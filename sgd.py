import numpy as np


def sigmoid(z):  # z = xT * w
	return 1 / (1 + np.exp(-z))


def f(x, w):
	return np.dot(x, w)  # or np.matmul
	# Careful with return value, it may be a matrix of 1 element [[0.4]]



def SGDSolver():
	pass