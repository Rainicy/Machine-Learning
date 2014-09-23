'''
Created May 27, 2014

@author Rainicy
'''

import numpy as np

from util import RMSE, initialData, loadData
from LinearBatchGD import LinearBatchGD
from LinearStochasticGD import LinearStochasticGD
from LogisticBatchGD import *
from LogisticStochasticGD import LogisticStochasticGD
from SmoothLogisticStochasticGD import SmoothLogisticStochasticGD


def main(i):

	# For testing
	np.set_printoptions(threshold='nan')

	# Part 1: Prepare for the data for training and testing
	data = np.loadtxt('../../data/spambase/spambase.data', delimiter=',')
	trainX, trainY, testX, testY = initialData(data,i)

	# Load the spam_polluted data.
	# trainX, trainY, testX, testY = loadData('../../data/spambase/spam_polluted/')


	# Part 2: Training the theta model by Batch Gradient Descent


	options = {'alpha': 5e-6, 'threshold': 1e-6, 'regularized': False, 'lambda': 50}
	# theta = LinearBatchGD(trainX, trainY, options)
	# theta = LinearStochasticGD(trainX, trainY, options)
	theta, rounds = LogisticBatchGD(trainX, trainY, options)
	# theta = LogisticStochasticGD(trainX, trainY, options)
	# theta = SmoothLogisticStochasticGD(trainX, trainY, options)

	# Part 3: Testing data 
	## for linear
	# predictTrain = np.dot(trainX, theta)
	# predictTest = np.dot(testX, theta)
	## for logistic 
	predictTrain = logistic(np.dot(trainX, theta))
	predictTest = logistic(np.dot(testX, theta))

	# rmseTrain = RMSE(predictTrain, trainY)
	# rmseTest = RMSE(predictTest, testY)

	# print "RMSE for Training Data: %f" % rmseTrain
	# print "RMSE for Testing Data: %f " % rmseTest


	# testing for logistic function
	predictTrain = np.squeeze(np.asarray(predictTrain))
	predictTest = np.squeeze(np.asarray(predictTest))
	trainER = np.sum((predictTrain > .5) != trainY) / float(trainY.size)
	testER = np.sum((predictTest > .5) != testY) / float(testY.size)
	print "Error Rate on training: %f" % trainER
	print "Error Rate on testing : %f" % testER

	return rounds, trainER, testER

if __name__ == '__main__':
	sum_rounds = 0
	sum_train = 0 
	sum_test = 0
	for i in range(10):
		rounds, train, test = main(i)
		sum_rounds += rounds
		sum_train += train
		sum_test += test

	print sum_rounds
	print sum_train
	print sum_test


