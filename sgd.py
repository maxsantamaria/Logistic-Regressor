import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 


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
		data = np.append(x, y, axis=1)
		np.random.shuffle(data)
		#x = data[:,:k+1]
		#y = data[:,k+1:]
		w = w - (alpha) * loss_derivative(x, y, w, lambd)

		loss_history.append(loss(x, y, w, lambd)[0][0])
		if loss_history[-1] < epsilon:
			break
	#print(loss_history)
	#print(w)
	return w, loss_history[-1]
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
		min_loss = 10**10
		best_param2 = w
		for alpha in np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 5):  # Logaritmic Scale
			#print('\nNEW LEARNING RATE ', alpha)
			for lambd in np.logspace(np.log10(lam_range[0]), np.log10(lam_range[1]), 5):
				#print('\tNEW LAMBDA ', lambd)
				w2, loss = SGD(x, y_aux, alpha, lambd, nepochs, 0, w)
				#print(loss)
				if loss < min_loss:
					min_loss = loss
					best_param = w2
					best_alpha = alpha
					best_lambd = lambd
		w2 = best_param

		#return w2, best_alpha, best_lambd

		y_aux = transform_y(y, 1)
		w1, loss = SGD(x, y_aux, best_alpha, best_lambd, 1000, 0, w)

		y_aux = transform_y(y, 0)
		w0, loss = SGD(x, y_aux, best_alpha, best_lambd, 1000, 0, w)

		return [w0, w1, w2], best_alpha, best_lambd

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


def SVMSolver(phase, x, y, clfs=[]):
	n = x.shape[0]
	x = normalize(x)
	x = np.hstack((np.array([1] * n)[:, np.newaxis], x))  # Add a column of 1s
	y = convert_y(y)
	if phase == "Training":
		clf2 = svm.SVC(kernel='linear')
		y_aux = transform_y(y, 2)
		clf2.fit(x, y_aux.ravel())

		clf1 = svm.SVC(kernel='linear')
		y_aux = transform_y(y, 1)
		clf1.fit(x, y_aux.ravel())

		clf0 = svm.SVC(kernel='linear')
		y_aux = transform_y(y, 0)
		clf0.fit(x, y_aux.ravel())

		print(clf2.predict(x[0, :].reshape(1, -1)))
		print(clf1.predict(x[0, :].reshape(1, -1)))
		print(clf0.predict(x[0, :].reshape(1, -1)))

		#clf = OneVsRestClassifier(svm.SVC(kernel='linear')).fit(x, y)
		#for i in range(n):
		#	print(clf0.predict(x[i, :].reshape(1, -1)), clf1.predict(x[i, :].reshape(1, -1)), clf2.predict(x[i, :].reshape(1, -1)), end=" ")
		#	print(y[i])
		return [clf0, clf1, clf2]
	elif phase == "Validation":
		total = 0
		for i in range(n):
			for index, clf in enumerate(clfs):
				prediction = clf.predict(x[i, :].reshape(1, -1))
				if prediction[0] == 1:
					break
			total += np.square(y[i] - prediction)
		return total / n

	elif phase == "Testing":
		prediction_array = np.zeros(y.shape)
		for i in range(n):
			for index, clf in enumerate(clfs):
				prediction = clf.predict(x[i, :].reshape(1, -1))
				if prediction[0] == 1:
					break
			prediction_array[i] = prediction
		
		print(confusion_matrix(y, prediction_array))
		print(accuracy_score(y, prediction_array))
		return prediction_array

		


	

def predict_SVM(clf):
	pass


if __name__ == '__main__':
	pass