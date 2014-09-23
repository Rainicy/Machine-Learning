'''
Created May 27, 2014

@author Rainicy
'''

import numpy as np

def loadData(folder):
	'''
	Description: This function is for loading the polled spam data from the given
				folder.
	'''
	trainX = np.loadtxt(folder + 'train_feature.txt', delimiter=' ')
	trainY = np.loadtxt(folder + 'train_label.txt', delimiter=' ')
	testX = np.loadtxt(folder + 'test_feature.txt', delimiter=' ')
	testY = np.loadtxt(folder + 'test_label.txt', delimiter=' ')
	## get mean and std values from training set
	mean = np.mean(trainX, axis = 0)
	std = np.std(trainX, axis = 0)
	trainX = (trainX - mean) / std
	testX = (testX - mean) / std
	
	return trainX, trainY, testX, testY

def initialData(data,i):
	'''
	Description: This function split the datta to training and testing set and 
				also split the features and labels. The last column is the label.
				Besides, we need shuffle the training data order.
				Partition the data follow the instruction on website:
				http://www.ccs.neu.edu/home/jaa/CS6140.13F/Homeworks/hw.03.html

	@param:
		data: The whole dataset.
	@return:
		trainX: training data features
		trainY: training data label
		testX: test data features
		testY: test data features
	'''
	# split training data and testing data
	# index end with digit 1 is the testing set, others are training set
	testIndex = np.arange(i, data.shape[0], 10)
	testData = data[testIndex]
	trainData = np.delete(data, testIndex, 0)

	# shffule the training data
	# np.random.shuffle(trainData)

	# get mean and std values
	mean = np.mean(trainData[:, :-1], axis = 0)
	std = np.std(trainData[:, :-1], axis = 0)
	# z-socre the features [(x-mean)/std]
	trainX = trainData[:, :-1]
	trainY = trainData[:, -1]
	trainX = (trainX - mean) / std
	testX = testData[:, :-1]
	testY = testData[:, -1]
	testX = (testX - mean) /std

	## flip first 400 labels in training
	# for i in range(400):
	# 	if trainY[i] == 0:
	# 		trainY[i] = 1
	# 	else:
	# 		trainY[i] = 0

	# add one column to the first column, which are all 1s
	trainX = np.insert(trainX, 0, 1, axis = 1)
	testX = np.insert(testX, 0, 1, axis = 1)
	return trainX, trainY, testX, testY

def RMSE(h, y):
	'''
	Description: Root Mean Squared Error(RMSE). J = sum(h - y)^2
				RMSE = sqrt(J/m). [m: #samples]
				Find more info on: 
				http://en.wikipedia.org/wiki/Root_mean_square_deviation

	@param:
		h: hypothese, calculated by (h = theta.T * X)
		y: the true label
	@return:
		RMSE: root mean Squared error
	'''
	h = np.squeeze(np.asarray(h))
	y = np.squeeze(np.asarray(y))
	J = np.sum((h - y)**2)
	return np.sqrt(J/y.size)
		

