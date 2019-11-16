from datetime import datetime

import mlrose
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

nn_model = mlrose.NeuralNetwork(
	hidden_nodes = [4],
	activation = 'relu',
	algorithm = 'genetic_alg',
	max_iters = 1000,
	is_classifier = True,
	# random_state = 27
)
nn_model.output_activation = "sigmoid"

nn_model.fit(x_train, y_train) # Training
y_pred = nn_model.predict(x_test) # Testing

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")
print("Report\n", classification_report(y_test, y_pred))
print("Execution time in seconds =", datetime.now() - startTime)
