#!/usr/bin/env python -W ignore
from datetime import datetime

import mlrose
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
							 confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

startTime = datetime.now()

dataset = np.loadtxt('EYES.csv', dtype=int, delimiter=',', skiprows=1)
y = dataset[:, -1]
x = dataset[:, 1: - 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

iters = 50

def model(algorithm, iterations=50, activation='relu'):
	nn_model = mlrose.NeuralNetwork(
		hidden_nodes = [4],
		activation = activation,
		algorithm = algorithm,
		max_iters = iterations,
		is_classifier = True,
		random_state = 72
	)
	nn_model.fit(x_train, y_train)
	y_pred = nn_model.predict(x_test)
	return y_pred

algos = {'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_desc'}

accuracies = {algo: accuracy_score(y_test, model(algo)) for algo in algos}

opt_algorithm = max(accuracies, key=lambda algo: accuracies[algo])

max_accuracy = accuracies[opt_algorithm]

print("Exploiting algorithm:", opt_algorithm)

same = 0

while same < 5 and max_accuracy < 98:
	iters += 550
	y_pred = model(opt_algorithm, iters)
	y_accuracy = accuracy_score(y_test, y_pred)

	if y_accuracy - max_accuracy < 1e-1: same += 1
	else: same = 0

	print(f"Current accuracy: {y_accuracy * 100}% In {iters} iterations")
	print(f"Current execution time elapsed = {datetime.now() - startTime}")

	if y_accuracy > max_accuracy: max_accuracy = y_accuracy


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")
print("Report\n", classification_report(y_test, y_pred))
print("Execution time in seconds =", datetime.now() - startTime)
