# Personalized Learning Platform

This project involves developing a personalized learning platform using various AI techniques, including neural networks, genetic algorithms, and first-order logic rules. The platform is designed to analyze large volumes of data related to individual performance, preferences, and learning styles to provide tailored learning experiences that adapt to students' needs in real-time.

## Project Overview

The program implements:
- A neural network with cross-validation and hyperparameter tuning to predict learning groups based on student performance data.
- A genetic algorithm to optimize learning paths by evaluating engagement and time spent on different modules.
- A set of first-order logic rules to determine student readiness for advanced topics.
- Visualization tools for data interpretation.

## Features

- **Neural Network Classifier**: Uses a multi-layer perceptron (MLP) with grid search for hyperparameter tuning to classify students into different learning groups.
- **Genetic Algorithm**: Optimizes learning paths by balancing engagement and time spent.
- **First-Order Logic (FOL) Rules**: Provides decision-making based on performance scores and prior knowledge.
- **Data Visualization**: Plots a confusion matrix to visualize the classifier's performance.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn
- DEAP (Distributed Evolutionary Algorithms in Python)

## Installation

1. Install Python 3.x if not already installed.
2. Install the required libraries using pip:

```bash
pip install numpy matplotlib scikit-learn deap

