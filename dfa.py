# import os
# try: os.chdir(os.path.join(os.getcwd(), '../../../../tmp'))
# except: pass
#%%
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

#%%
def create_dataset(n_in, n_out, len_samples):
	'''Creates randomly a matrix n_out x n_in which will be
	   the target function to learn. Then generates
	   len_samples examples which will be the training set'''
	M = np.random.randint(low=-10, high=10, size=(n_out, n_in))
	samples = []
	targets = []
	for i in range(len_samples):
		sample = np.random.randn(n_in)
		samples.append(sample)
		targets.append(np.dot(M, sample))

	return M, np.asarray(samples), np.asarray(targets)

#%%
def forward_pass(input, W1, W2):
	a1 = np.dot(W1, input)
	y_hat = np.dot(W2, a1)
	return a1, y_hat

def backward_pass(e, B1, input, a1):
	dW1 = -(np.dot(np.dot(B1, e), np.transpose(input)))
	dW2 = -np.dot(e, np.transpose(a1))

	return dW1, dW2

#%%
def average_angle(W2, B1, error):
	dh1 = np.mean(np.dot(B1, error), axis=1)
	c1 = np.mean(np.dot(np.transpose(W2), error), axis=1)
	dh1_norm = np.linalg.norm(dh1)
	c1_norm = np.linalg.norm(c1)
	inverse_dh1_norm = np.power(dh1_norm, -1)
	inverse_c1_norm = np.power(c1_norm, -1)

	Lk = np.matmul(np.transpose(dh1), c1)*inverse_dh1_norm*inverse_c1_norm
	beta = np.arccos(np.clip(Lk, -1., 1.))*180/np.pi
	return Lk, beta

#%%
def train_on_dataset(samples, targets, n_in, n_out, n_hidden, n_epoch=4000, lr=1e-6, tol=1e-3):
	W1 = np.zeros((n_hidden, n_in))
	W2 = np.zeros((n_out, n_hidden))

	B1 = np.random.randn(n_hidden, n_out)

	angles = []
	Lk = []

	samples = np.transpose(samples)
	targets = np.transpose(targets)

	for i in range(n_epoch):
		a1, y_hat = forward_pass(samples, W1, W2)
		errors = y_hat - targets
		cost = 0.5*np.sum((y_hat-targets)**2)
		dW1, dW2 = backward_pass(errors, B1, samples, a1)
		W1 += lr*dW1
		W2 += lr*dW2
		# every 50 epochs excpet for the first one (weights are zero
		# and norms go to zero -> numerical instability)
		if (i-1)%10==0:
			print('Cost:', cost)
			print('Computing angle between updates - epoch', i)
			crit, beta = average_angle(W2, B1, errors)
			angles.append(beta)
			Lk.append(crit)
			print('Alignment criterion:', crit, '> 0', crit > 0, '\n')
		if cost <= tol:
			break
	return W1, W2, angles, Lk

#%%
M, samples, targets = create_dataset(10, 10, 1000)
w1, w2, angles, Lk = train_on_dataset(samples, targets, 10, 10, 1000)

#%%
O = np.dot(w2, w1)
print('D:', np.sum(np.abs(M - O)))

#%%
plt.plot(range(len(angles)), angles)
plt.xlabel('Iterations/50')
plt.ylabel('Angle (degrees)')
plt.xlim([0, range(len(angles))[-1]])
plt.ylim([0, 100])

#%%
plt.plot(range(len(Lk)), Lk)
plt.xlim([0, range(len(Lk))[-1]])
plt.xlabel('Iterations/50')
plt.ylabel('Alignment criterion')
