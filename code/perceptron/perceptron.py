'''
Created on May 30, 2014

@author Rainicy
'''

import numpy as np

from util import RMSE

def percep(n):
	'''
	Description: To calculate the perceptron of the given vector.
				if n >= 0, return 1, otherwise return 0.

	@param:
		n: given a vector
	@return:
		logistic: return the sigmoid function based on the given vector 
	'''

	n[n>=0] = 1
	n[n<0] = 0
	return n
		
def perceptron(X, y, alpha, threshold):
	'''
	Description: This algorithms represents the Logistic Batch Gradient Descent algorithm.

	@param:
		X: training features
		y: training labels
		alpha: learning rate
		threshold: the threshold for terminate the loop
	@return:
		theta: the parameters model
	'''

	# m: #samples, n:#features
	m, n = np.shape(X)
	# intialize the theta with all 0s
	theta = np.zeros(n)
	loop = 0

	# initialize the RMSE for terminating the loop
	hypothese = percep(np.dot(X, theta))
	rmse = RMSE(hypothese, y)
	rmse_ = np.inf
	xTrans = np.transpose(X)

	# stop looping condition
	while abs(rmse - rmse_) > threshold:
		loop += 1
		if loop == 1:
			rmse_ = np.inf
		else:
			rmse_ = rmse

		# updating parameters
		theta = theta + alpha * np.dot(xTrans, (y - hypothese))

		hypothese = percep(np.dot(X, theta))
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f' % (loop, rmse)

	return theta