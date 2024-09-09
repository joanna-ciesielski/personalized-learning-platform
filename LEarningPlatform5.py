import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from deap import base, creator, tools, algorithms

# Sample dataset: Student Performance Data
data = np.array([
    [85, 80], [70, 75], [60, 65], [90, 95], [50, 55], [80, 82],
    [40, 45], [92, 85], [72, 78], [68, 70], [88, 90], [55, 60],
    [65, 68], [78, 82], [35, 40], [95, 93], [85, 88], [45, 48]
])
targets = np.array([1, 2, 2, 1, 3, 1, 3, 1, 2, 2, 1, 3, 2, 1, 3, 1, 1, 3])

# Combine the data and targets to facilitate resampling
dataset = np.column_stack((data, targets))

# Separate majority and minority classes
class_1 = dataset[dataset[:, -1] == 1]
class_2 = dataset[dataset[:, -1] == 2]
class_3 = dataset[dataset[:, -1] == 3]

# Oversample minority classes
class_2_upsampled = resample(class_2, replace=True, n_samples=len(class_1), random_state=42)
class_3_upsampled = resample(class_3, replace=True, n_samples=len(class_1), random_state=42)

# Combine majority class with upsampled minority classes
upsampled_dataset = np.vstack((class_1, class_2_upsampled, class_3_upsampled))

# Separate back into features and targets
X_upsampled = upsampled_dataset[:, :-1]
y_upsampled = upsampled_dataset[:, -1]

# Normalize the features
scaler = StandardScaler()
X_upsampled = scaler.fit_transform(X_upsampled)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)

# Improved Neural Network with Cross-Validation and Hyperparameter Tuning
def train_neural_network(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(5,), (10,), (5, 5), (10, 5)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [2000, 3000],
        'solver': ['adam', 'sgd']
    }
    nn = MLPClassifier(random_state=42)
    grid_search = GridSearchCV(nn, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f'Best Parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_

# Train the neural network
nn_model = train_neural_network(X_train, y_train)

# Predict learning group for test data
y_pred = nn_model.predict(X_test)
print(f'Neural Network Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred, zero_division=0))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Enhanced Genetic Algorithm for Learning Path Optimization
learning_paths = np.array([
    [0.9, 0.8, 0.85, 0.7],  # Learning Path 1
    [0.75, 0.65, 0.6, 0.7], # Learning Path 2
    [0.8, 0.9, 0.85, 0.95], # Learning Path 3
    [0.6, 0.55, 0.5, 0.4],  # Learning Path 4
    [0.95, 0.9, 0.85, 0.8]  # Learning Path 5
])

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    # Refined fitness function considering engagement and time spent
    engagement_score = sum(individual)
    time_spent = len(individual) * 0.5  # Assuming each module takes 0.5 units of time
    return engagement_score - time_spent,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Increasing population size and generations
population = toolbox.population(n=20)
NGEN = 20
CXPB, MUTPB = 0.5, 0.2

for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = list(map(toolbox.evaluate, offspring))
    
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    population = toolbox.select(offspring, k=len(population))

best_solution = tools.selBest(population, k=1)[0]
print(f'Best Learning Path Solution: {best_solution}')

# First-Order Logic Rules
def check_readiness(performance_score, prior_knowledge):
    if performance_score > 80 and prior_knowledge > 70:
        return "Ready for Advanced Topics"
    elif performance_score > 70:
        return "Ready for Intermediate Topics"
    else:
        return "Need More Practice"

# Example of FOL-based decision
for student in data:
    result = check_readiness(student[0], student[1])
    print(f'Student with knowledge {student[0]} and score {student[1]}: {result}')

# Visualization
def plot_confusion_matrix(cm):
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)
