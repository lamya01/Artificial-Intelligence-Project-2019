import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

dataset = np.loadtxt('EYES.csv', dtype=int, delimiter=',', skiprows=1)
labels = dataset[:,  - 1]
data = dataset[:, 1: - 1]

X_train, X_val, Y_train, Y_val = train_test_split(
	data, labels, stratify=labels, random_state=0)


class FeedForwardNeuralNet:
	def __init__(self, n_inputs, hidden_sizes):
		self.nx = n_inputs
		self.ny = 1
		self.nh = len(hidden_sizes)
		self.sizes = [self.nx] + hidden_sizes + [self.ny]

		self.W = {
			i + 1: np.random.randn(self.sizes[i], self.sizes[i + 1]) for i in range(self.nh + 1)}
		self.B = {i + 1: np.zeros((1, self.sizes[i + 1]))
					for i in range(self.nh + 1)}

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def grad_sigmoid(self, x):
		return x * (1 - x)

	def relu(self, x):
		return max(0, x)

	def forward_pass(self, x):
		self.A = {}
		self.H = {}
		self.H[0] = x.reshape(1, -1)
		for i in range(self.nh + 1):
			self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1]) + self.B[i + 1]
			self.H[i + 1] = self.sigmoid(self.A[i + 1])
		return self.H[self.nh + 1]

	def grad(self, x, y):
		self.forward_pass(x)
		self.dW = {}
		self.dB = {}
		self.dH = {}
		self.dA = {}
		L = self.nh + 1
		self.dA[L] = (self.H[L] - y)
		for k in range(L, 0,  - 1):
			self.dW[k] = np.matmul(self.H[k - 1].T, self.dA[k])
			self.dB[k] = self.dA[k]
			self.dH[k - 1] = np.matmul(self.dA[k], self.W[k].T)
			self.dA[k - 1] = np.multiply(self.dH[k - 1],
										 self.grad_sigmoid(self.H[k - 1]))

	def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True):
		if initialise:
			for i in range(self.nh + 1):
				self.W[i +
						 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
				self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))

		for e in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
			dW = {}
			dB = {}
			for i in range(self.nh + 1):
				dW[i + 1] = np.zeros((self.sizes[i], self.sizes[i + 1]))
				dB[i + 1] = np.zeros((1, self.sizes[i + 1]))
			for x, y in zip(X, Y):
				self.grad(x, y)
				for i in range(self.nh + 1):
					dW[i + 1] += self.dW[i + 1]
					dB[i + 1] += self.dB[i + 1]

			m = X.shape[1]
			for i in range(self.nh + 1):
				self.W[i + 1] -= learning_rate * dW[i + 1] / m
				self.B[i + 1] -= learning_rate * dB[i + 1] / m

	def predict(self, X):
		Y_pred = [self.forward_pass(x) for x in X]
		return np.array(Y_pred).squeeze()

	def getWeights(self):
		return self.W, self.B, self.A, self.H, self.dW, self.dB, self.dA, self.dH

ffnn = FeedForwardNeuralNet(24, [4])
ffnn.fit(X_train, Y_train, epochs=1000, learning_rate=.001)

Y_pred_train = ffnn.predict(X_train)
Y_pred_binarised_train = (Y_pred_train >= 0.5).ravel()
Y_pred_val = ffnn.predict(X_val)
Y_pred_binarised_val = (Y_pred_val >= 0.5).ravel()
accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)
accuracy_val = accuracy_score(Y_pred_binarised_val, Y_val)

print(Y_pred_train)

print(f"Training accuracy {accuracy_train: .2f}")
print(f"Validation accuracy {accuracy_val: .2f}")

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_pred_binarised_train,
			s=15 * (np.abs(Y_pred_binarised_train - Y_train) + .2))
plt.show()
