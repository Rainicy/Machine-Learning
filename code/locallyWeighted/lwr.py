'''
Created on May 30, 2014,

@author Rainicy
'''

import numpy as np

def gaussianKernel(xi, x, k, c=1.0):
	'''
	Description: Gaussian Kernel.
				Reference: 
				http://vilkeliskis.com/blog/2013/09/08/machine_learning_part_2_locally_weighted_linear_regression.html

	@param:
		xi: the nearby point, from one of the training data set
		x: the test sample point
		c, k : Gaussian kernel parameters

	@return:
		w: the weight on the position ith in the training datas set. 
	'''
	diff = x - xi
	product = diff * diff.T
	return c * np.exp(product / (-2.0 * (k ** 2)))

def lwr(trainX1, trainY1, testX, k=1.0):
	'''
	Description: Predict a test sample, by fitting local weighted regression.

	@param:
		trainX: training data features [m * n]
		trainY: training data label [m * 1]
		testX: one simple testing sample [n * 1]
		k: kernel function parameter 
	@return:
		y: predict value by the given test sample
	'''

	trainX = np.mat(trainX1)
	trainY = np.mat(trainY1).T
	# m: #samples, n:#features
	m, n = np.shape(trainX)

	# initial weights
	weights = np.mat(np.eye(m))
	# update weights 
	for i in range(m):
		weights[i, i] = gaussianKernel(trainX[i, :], testX, k)

	# get the theta 
	# the rules in note page 3-4
	xTx = trainX.T * (weights * trainX)
	# if np.linalg.det(xTx) == 0.0:
	# 	print "This matrix is singular, non-invertible"
	# 	return 

	# calculate theta 
	theta = xTx.I * (trainX.T * (weights * trainY))

	# return predict test point value
	return testX * theta
	