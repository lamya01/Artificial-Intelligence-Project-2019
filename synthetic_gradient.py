# from os import chdir
# chdir('Projects/AI')

import numpy as np
from sklearn.preprocessing import StandardScaler


def sigmoid(x): return 1 / (1 + np.exp(-x))

def grad_sigmoid(x): return x * (1 - x)

def relu(x): return np.maximum(0, x)

def grad_relu(x): return np.array([[1 if i else 0 for i in j] for j in x])


dataset = np.loadtxt('EYES.csv', dtype=int, delimiter=',', skiprows=1)
y = dataset[:, -1]
x = dataset[:, 1: - 1]

scaler = StandardScaler()
x = scaler.fit_transform(x)

inp_dim = x.shape[1]
layer_1_dim = 4
layer_2_dim = 4
out_dim = 2

iterations = 1000
batch_size = 2


class DNI:
	def __init__(self, inp_dim, out_dim, activation, activation_deriv, alpha=0.1):
		self.weights = (np.random.randn(inp_dim, out_dim) * 0.2) - 0.1
		self.activation = activation
		self.activation_deriv = activation_deriv

		self.weights_synthetic_grads = (np.random.randn(out_dim, out_dim) * 0.2) - 0.1
		self.alpha = alpha

	def forward_and_synthetic_update(self, inp):
		self.inp = inp
		# forward propagate
		self.out = self.activation(self.inp.dot(self.weights))
		# generate synthetic gradient via simple linear transformation
		self.synthetic_gradient = self.out.dot(self.weights_synthetic_grads)
		# update our regular weights using synthetic gradient
		self.weight_synthetic_gradient = self.synthetic_gradient * self.activation_deriv(self.out)

		self.weights += self.inp.T.dot(self.weight_synthetic_gradient) * self.alpha
		# return backpropagated synthetic gradient (this is like the out of "backprop" method from the Layer class)
		# also return forward propagated out
		return self.weight_synthetic_gradient.dot(self.weights.T), self.out

	# this is just like the "update" method from before... except it operates on the synthetic weights
	def update_synthetic_weights(self, true_gradient):
		self.synthetic_gradient_delta = self.synthetic_gradient - true_gradient
		self.weights_synthetic_grads += self.out.T.dot(self.synthetic_gradient_delta) * self.alpha


layer_1 = DNI(inp_dim, layer_1_dim, relu, grad_relu)
layer_2 = DNI(layer_1_dim, layer_2_dim, relu, grad_relu)
layer_3 = DNI(layer_2_dim, out_dim, sigmoid, grad_sigmoid)

for iter in range(iterations):
	error = 0

	for batch_i in range(len(x) // batch_size):
		batch_x = x[(batch_i * batch_size):(batch_i + 1) * batch_size]
		batch_y = y[(batch_i * batch_size):(batch_i + 1) * batch_size]

		layer_1_synthetic_grad, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
		layer_2_synthetic_grad, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
		layer_3_synthetic_grad, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)

		layer_1.update_synthetic_weights(layer_1_synthetic_grad)
		layer_2.update_synthetic_weights(layer_2_synthetic_grad)
		layer_3.update_synthetic_weights(layer_3_synthetic_grad)

print(layer_3_out)
