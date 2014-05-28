'''
Created on May 27, 2014

@author Rainicy
'''

import numpy as np
import pandas as pd


		
def initialData(data):
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
	testIndex = np.arange(1, data.shape[0], 10)
	testData = data[testIndex]
	trainData = np.delete(data, testIndex, 0)

	# shffule the training data
	np.random.shuffle(trainData)

	# get mean and std values
	mean = np.mean(data[:, :-1], axis = 0)
	std = np.std(data[:, :-1], axis = 0)
	# z-socre the features [(x-mean)/std]
	trainX = trainData[:, :-1]
	trainY = trainData[:, -1]
	trainX = (trainX - mean) / std
	testX = testData[:, :-1]
	testY = testData[:, -1]
	testX = (testX - mean) /std

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
		J = np.sum((h - y)**2)
		return np.sqrt(J/y.size)
		

def BatchGD(X, y, alpha=5e-5, threshold=1e-3):
	'''
	Description: This algorithms represents the Batch Gradient Descent algorithm.

	@param:
		X: training features
		y: training labels
		alpha: learning rate
		threshold: the threshold for terminate the loop
	@return:
		theta: the parameters, which have been trained for feture testing
	'''

	# m: #samples, n:#features
	m, n = np.shape(X)
	# intialize the theta with all 0s
	theta = np.zeros(n)
	loop = 0

	# initialize the RMSE for terminating the loop
	hypothese = np.dot(X, theta)
	rmse = RMSE(hypothese, y)
	rmse_ = np.inf
	xTrans = np.transpose(X)
	while abs(rmse - rmse_) > threshold:
		loop += 1
		if loop == 1:
			rmse_ = np.inf
		else:
			rmse_ = rmse
		hypothese = np.dot(X, theta)
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f' % (loop, rmse)

		# updating parameters
		theta = theta + alpha * np.dot(xTrans, (y - hypothese))

	return theta



def main():

	np.set_printoptions(threshold='nan')

	# Part 1: Prepare for the data for training and testing

	data = np.loadtxt('../data/spambase/spambase.data', delimiter=',')
	trainX, trainY, testX, testY = initialData(data)

	# Part 2: Training the theta model by Batch Gradient Descent
	alpha = 5e-5
	threshold = 1e-6
	theta = BatchGD(trainX, trainY, alpha, threshold)

	# Part 3: Testing data 
	predictTrain = np.dot(trainX, theta)
	predictTest = np.dot(testX, theta)

	rmseTrain = RMSE(predictTrain, trainY)
	rmseTest = RMSE(predictTest, testY)

	print "RMSE for Training Data: %f", rmseTrain
	print "RMSE for Testing Data: %f", rmseTest

if __name__ == '__main__':
	main()