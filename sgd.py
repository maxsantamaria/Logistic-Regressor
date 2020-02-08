import numpy as np


def sigmoid(z):  # z = xT * w
	return 1 / (1 + np.exp(-z))


def z(x, w):
	return np.dot(x, w)  # or np.matmul
	# Careful with return value, it may be a matrix of 1 element [[0.4]]


def loss(x, y, w, lambd):
	h = sigmoid(z(x, w))
	h[h == 1] = 0.999
	n = x.shape[0]
	return (1/n) * (np.dot((-y).T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) + lambd * np.dot(w.T, w)


def loss_derivative(x, y, w, lambd):
	h = sigmoid((z(x, w)))
	n = x.shape[0]
	return np.dot(x.T, (h - y)) + lambd * 2 * w


def MSE(x, y, ws):
	predicted_y = predict(x, ws).reshape(-1, 1)
	diff = np.square(y - predicted_y)
	mse = diff.mean()
	return mse



def predict(x, params):  # Params in order [w0, w1, w2]
	predictions = []
	for w in params:
		predictions.append(sigmoid(z(x, w)))
	predicted_y = np.zeros(x.shape[0])
	for sample in range(predictions[0].shape[0]):
		max_index = 0
		max_prob = 0
		for index in range(len(predictions)):
			#print(predictions[index][sample])
			if predictions[index][sample] > max_prob:
				max_index = index
				max_prob = predictions[index][sample]
		predicted_y[sample] = max_index
		
	#for i in range(len(predictions[0])):
	#	print(predictions[0][i], predictions[1][i], predictions[2][i])
		
	return predicted_y

	#return np.round(sigmoid(z(x, w)))  # 0 or 1


def SGD(x, y, alpha, lambd, nepoch, epsilon, w):
	n = x.shape[0]
	k = x.shape[1] - 1
	loss_history = [loss(x, y, w, lambd)[0][0]]
	w_history = []
	for i in range(nepoch):
		w = w - (alpha) * loss_derivative(x, y, w, lambd)
		loss_history.append(loss(x, y, w, lambd)[0][0])
	#print(loss_history[-1])
	#print(w)
	return w
	#print(predict(x[11, :], w))



def SGDSolver(phase, x, y, alpha_range=0, lam_range=0, nepochs=1000, epsilon=0.001, w=0):
	#x, w = best_correlation(x, w) LATER
	n = x.shape[0]
	x = normalize(x)
	x = np.hstack((np.array([1] * n)[:, np.newaxis], x))  # Add a column of 1s

	y = convert_y(y)
	if phase == "Training":
		w = np.array(w).reshape(-1, 1)
		y_aux = transform_y(y, 2)
		#print(loss(sigmoid(f(x, w)), y))
		#print(loss(x, y, w))
		w2 = SGD(x, y_aux, 0.01, 0.00001, 1000, 0, w)
		
		y_aux = transform_y(y, 1)
		w1 = SGD(x, y_aux, 0.01, 0.00001, 1000, 0, w)

		y_aux = transform_y(y, 0)
		w0 = SGD(x, y_aux, 0.01, 0.00001, 1000, 0, w)

		return [w0, w1, w2], 0.01, 0.00001

		predicted_y = predict(x, [w0, w1, w2])
		#for i in range(len(predicted_y)):
		#	print(predicted_y[i], y[i])
	elif phase == "Validation":
		return MSE(x, y, w)
	elif phase == "Testing":
		return predict(x, w)




def normalize(x):
	n = x.shape[0]
	k = x.shape[1]
	for i in range(k):
		feature = x[:, i]
		min_x = min(feature)
		max_x = max(feature)
		feature = (feature - min_x) / (max_x - min_x)
		feature = feature - np.mean(feature)
		x[:, i] = feature
	return x


def convert_y(y):
	new_y = np.zeros(y.shape)
	for i in range(len(y)):
		if y[i] >= 7:
			new_y[i] = 2
		elif y[i] <= 4:
			new_y[i] = 0
		else:
			new_y[i] = 1
	return new_y
			

def transform_y(y, class_number):
	new_y = np.zeros(y.shape)
	for i in range(len(y)):
		if y[i] == class_number:
			new_y[i] = 1
		else:
			new_y[i] = 0	
	return new_y


def best_correlation(x, w):
	pairs = []
	delete = set()
	best_correlations = {}
	for i in range(x.shape[1]):
		for j in range(x.shape[1]):
			if i != j and (i , j) not in pairs:
				corr = np.corrcoef(x[:, i], x[:, j])
				pairs.append((j, i))
				if abs(corr[0][1]) >= 0.55: 
					#print("eliminar", i, j)
					delete.add(j)
	for j in delete:
		x = np.delete(x, j, 1)
		w = np.delete(w, j).reshape(-1, 1)
	return x, w




if __name__ == '__main__':
	pass