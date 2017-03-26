import numpy as np 
import scipy.misc
import pandas as pd 
import sklearn
from sklearn.cross_validation import train_test_split

def data_parser():

	trainX = np.load('tinyX.npy')
	trainY = np.load('tinyY.npy')
	
	trainX = [np.reshape(x, (12288, 1)) for x in trainX]

	trainX = np.asarray(trainX)

	results = [vectorized(y) for y in trainY]

	results = np.asarray(results)

	X_train, X_test, y_train, y_test = train_test_split(trainX, results, test_size=0.2, random_state=0)

	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)

	training_data = zip(X_train, y_train)

	X_test = np.asarray(X_test)
	y_test = np.asarray(y_test)

	y_test_final = []
	for x in y_test:
		y_test_final.append(np.argmax(x))

	y_test_final = np.asarray(y_test_final)

	validation_data = zip(X_test, y_test_final)

	return training_data, validation_data

def vectorized(y):
	temp = np.zeros(((40,1)), dtype=np.int8)
	temp[y] = 1.0
	return temp 

#scipy.misc.imshow(trainY[0].transpose(2,1,0)) # put RGB channels last