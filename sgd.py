import numpy as np


def sigmoid(z):  # z = xT * w
	return 1 / (1 + np.exp(-z))


def z(x, w):
	return np.dot(x, w)  # or np.matmul
	# Careful with return value, it may be a matrix of 1 element [[0.4]]


def loss(x, y, w):
	h = sigmoid(z(x, w))
	h[h == 1] = 0.999
	return (np.dot((-y).T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h)))


def loss_derivative(x, y, w):
	h = sigmoid((z(x, w)))
	return np.dot(x.T, (h - y))


def predict(x, w):
	return np.round(sigmoid(z(x, w)))  # 0 or 1


def SGD(x, y, alpha, lambd, nepoch, epsilon, w):
	n = x.shape[0]
	k = x.shape[1] - 1
	loss_history = [loss(x, y, w)]
	w_history = []
	for i in range(nepoch):
		w = w - (alpha) * loss_derivative(x, y, w)
		#print(w)
		#print((z(x, w)))
		
		loss_history.append(loss(x, y, w)[0][0])

	print(loss_history)
	#print(predict(x[11, :], w))
	#print(y[11])



def SGDSolver(x, w, y):
	n = x.shape[0]
	x = np.hstack((np.array([1] * n)[:, np.newaxis], x))  # Add a column of 1s
	y = convert_y(y)
	y_aux = transform_y(y, 2)
	
		

	#print(loss(sigmoid(f(x, w)), y))
	#print(loss(x, y, w))
	SGD(x, y_aux, 0.00001, 0, 100, 0, w)


def convert_y(y):
	for i in range(len(y)):
		if y[i] >= 7:
			y[i] = 2
		elif y[i] <= 4:
			y[i] = 0
		else:
			y[i] = 1
	return y
			

def transform_y(y, class_number):
	new_y = np.zeros(y.shape)
	for i in range(len(y)):
		if y[i] == class_number:
			new_y[i] = 1
		else:
			new_y[i] = 0	
	return new_y


if __name__ == '__main__':
	pass