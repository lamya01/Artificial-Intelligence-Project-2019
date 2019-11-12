import os

import mlrose
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

os.chdir(os.path.join(os.getcwd(), 'Projects/AI/'))
print(os.getcwd())

# %%
dataset = np.loadtxt('EYES.csv', dtype=int, delimiter=',', skiprows=1)
labels = dataset[:, -1]
data = dataset[:, 1: - 1]
print(data.shape, labels.shape)

# %%
x_train, x_test, y_train, y_test = train_test_split(
	data, labels, stratify=labels, random_state=0)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# %%
scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[4], activation='relu', algorithm='random_hill_climb', max_iters=1000, bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=5, max_attempts=100, random_state=0)

nn_model1.fit(x_train_scaled, y_train_hot)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(x_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print('Training accuracy: ', y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(x_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print('Test accuracy: ', y_test_accuracy)
