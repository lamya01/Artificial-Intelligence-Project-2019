#!/usr/bin/env python3
'''Authors
Sarthak Gupta (PES1201700077)
Dhruv Vohra   (PES1201700281)
Lamya Bhasin  (PES1201701244)
'''

from datetime import datetime

import mlrose
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings; warnings.filterwarnings("ignore")

startTime = datetime.now()

dataset = np.loadtxt('EYES.csv', dtype=int, delimiter=',', skiprows=1)
y = dataset[:, -1]
x = dataset[:, 1: - 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

def model(algorithm, iterations=50, activation='relu'):
	nn = mlrose.NeuralNetwork(
		hidden_nodes = [4],
		activation = activation,
		algorithm = algorithm,
		max_iters = iterations,
		is_classifier = True,
		random_state = 2
	)
	nn.output_activation = mlrose.activation.sigmoid
	nn.fit(x_train, y_train)
	y_pred = nn.predict(x_test)
	return y_pred

algos = ('random_hill_climb', 'simulated_annealing', 'gradient_descent') # , 'genetic_alg',

[print(f"Exploring {algo}") for algo in algos]
accuracies = {algo: accuracy_score(y_test, model(algo)) for algo in algos}

opt_algorithm = max(accuracies, key=lambda algo: accuracies[algo])

y_accuracy = accuracies[opt_algorithm]

print(f"Exploiting {opt_algorithm} algorithm")

y_pred = model(opt_algorithm, 1000)
y_accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")
print("Report\n", classification_report(y_test, y_pred))
print("Execution time in seconds =", datetime.now() - startTime)
