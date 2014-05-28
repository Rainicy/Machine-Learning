'''
Created May 27, 2014

@author Rainicy
'''

import numpy as np

from LinearBatchGD import LinearBatchGD
from LinearStochasticGD import LinearStochasticGD
from util import RMSE, initialData


def main():

	# For testing
	np.set_printoptions(threshold='nan')

	# Part 1: Prepare for the data for training and testing

	data = np.loadtxt('../../data/spambase/spambase.data', delimiter=',')
	trainX, trainY, testX, testY = initialData(data)

	# Part 2: Training the theta model by Batch Gradient Descent
	alpha = 5e-5
	threshold = 1e-6
	theta = LinearBatchGD(trainX, trainY, alpha, threshold)
	theta = LinearStochasticGD(trainX, trainY, alpha, threshold)

	# Part 3: Testing data 
	# predictTrain = np.dot(trainX, theta)
	# predictTest = np.dot(testX, theta)

	# rmseTrain = RMSE(predictTrain, trainY)
	# rmseTest = RMSE(predictTest, testY)

	# print "RMSE for Training Data: %f" % rmseTrain
	# print "RMSE for Testing Data: %f " % rmseTest

if __name__ == '__main__':
	main()