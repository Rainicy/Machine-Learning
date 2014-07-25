'''
Created on July 24, 2014

@author Rainicy
'''

import numpy as np

def initialData(data):
	'''
	Description: This function split the datta to training and testing set and 
				also split the features and labels. The last column is the label.
				Besides, we need shuffle the training data order.
				Partition the data follow the instruction on website:
				http://www.ccs.neu.edu/home/jaa/CS6140.13F/Homeworks/hw.02.html

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
	testIndex = np.arange(1, data.shape[0], 10)
	testData = data[testIndex]
	trainData = np.delete(data, testIndex, 0)

	# shffule the training data
	np.random.shuffle(trainData)

	# get mean values
	# Pr[fi <= mui | spam]
	# Pr[fi > mui | spam]
	# Pr[fi <= mui | non-spam]
	# Pr[fi > mui | non-spam]
	# transform all fi = 1, if fi>mui, otherwise fi = 0. 
	mean = np.mean(data[:, :-1], axis = 0)

	trainX = trainData[:, :-1]
	trainX[trainX <= mean] = 0
	trainX[trainX > mean] = 1
	trainY = trainData[:, -1]

	testX = testData[:, :-1]
	testX[testX <= mean] = 0
	testX[testX > mean] = 1
	testY = testData[:, -1]

	return trainX, trainY, testX, testY
