'''
Created on July 24, 2014

@author Rainicy
'''

import numpy as np
from random import *

def loadData(train, test):
	'''
	Description: Given the train and test files names, and load the data.

	@return:
		trainX:	
		trainY:
		testX:
		testY:
	'''
	trainData = np.loadtxt(train, delimiter=',')
	mean = np.mean(trainData[:, :-1], axis = 0)
	testData = np.loadtxt(test, delimiter=',')

	trainX = trainData[:, :-1]
	index1 = (trainX > mean) & (trainX != -1)
	index2 = (trainX <= mean) & (trainX != -1)
	trainX[index1] = 1
	trainX[index2] = 0
	trainY = trainData[:, -1]

	testX = testData[:, :-1]
	index1 = (testX > mean) & (testX != -1)
	index2 = (testX <= mean) & (testX != -1)
	testX[index1] = 1
	testX[index2] = 0
	testY = testData[:, -1]

	return trainX, trainY, testX, testY

def splitData(data):
	'''
	Description: Split the data into training and testing part, including 80% training,
					and 20% testing.

	@param:
		data: The whole dataset.
	@output:
		files: 
			1) p_percent_missing_train.txt
			2) p_percent_missing_test.txt
	'''

	outFolder = "../../data/spambase/missing_values/"

	m, n = data.shape

	total = m*(n-1)
	## creates different percent missing values data from 0%~90%
	for percent in np.arange(0, 10, 1):
		print "Generating {}0% missing feature".format(percent)
		miss_count = 0
		for i in range(m):
			for j in range(n-1): ## without change the label
				if randint(1,10) <= percent:
					miss_count += 1
					data[i][j] = -1

		print "Missing: {}, Actual Percent: {:.2f}".format(miss_count, float(miss_count)/total)

		## split the data to training and testing
		# folder 1&2 as testing dataset, and the rest of them as training set.
		index_1 = np.arange(1, m, 10)
		index_2 = index_1 + 1
		index_1_2 = np.concatenate((index_1,index_2), axis = 0)
		testData = data[index_1_2]
		trainData = np.delete(data, index_1_2, 0)

		test = np.asarray(testData)
		train = np.asarray(trainData)
		np.savetxt(outFolder + str(percent*10) + "_percent_missing_test.txt", test, delimiter=",", fmt='%f')
		np.savetxt(outFolder + str(percent*10) + "_percent_missing_train.txt", train, delimiter=",", fmt='%f')



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

	# get mean values
	# Pr[fi <= mui | spam]
	# Pr[fi > mui | spam]
	# Pr[fi <= mui | non-spam]
	# Pr[fi > mui | non-spam]
	# transform all fi = 1, if fi>mui, otherwise fi = 0. 
	mean = np.mean(data[:, :-1], axis = 0)

	trainX = trainData[:, :-1]
	index1 = trainX > mean
	index2 = trainX <= mean
	trainX[index1] = 1
	trainX[index2] = 0
	trainY = trainData[:, -1]

	testX = testData[:, :-1]
	index1 = testX > mean
	index2 = testX <= mean
	testX[index1] = 1
	testX[index2] = 0
	testY = testData[:, -1]

	return trainX, trainY, testX, testY
