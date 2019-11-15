from datetime import datetime

import mlrose
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
os.chdir(os.path.join(os.getcwd(), '/home/sarthak7gupta/batcave/Semester5/Projects/AI/'))
print(os.getcwd())

startTime = datetime.now()

dataset = np.loadtxt('EYES.csv', dtype=int, delimiter=',', skiprows=1)
y = dataset[:, -1]
x = dataset[:, 1: - 1]
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# 'random_hill_climb'
# 'simulated_annealing'
# 'genetic_alg'
# ''

nn_model = mlrose.NeuralNetwork(
	hidden_nodes = [12],
	activation = 'relu',
	algorithm = 'genetic_alg',
	max_iters = 1,
	is_classifier = True,
	learning_rate = 0.001,
	max_attempts = 100,
	random_state = 15
)


nn_model.fit(x_train, y_train)

y_pred = nn_model.predict(x_test)

y_pred_labels = [1 if i else 0 for i in list(y_pred > 0.9)]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))
print(f"Accuracy: {accuracy_score(y_test, y_pred_labels) * 100}%")
print("Report\n", classification_report(y_test, y_pred_labels))
print("Execution time in seconds =", datetime.now() - startTime)
