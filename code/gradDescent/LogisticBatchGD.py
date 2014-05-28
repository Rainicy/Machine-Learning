'''
Created on May 27, 2014

@author Rainicy
'''

import numpy as np
from util import RMSE

def logistic(n):
	'''
	Description: To calculate the logistic of the given number. Also called Sigmoid function.
				 Calculated by (logistic = 1 / (1 + exp(-n))).
				 Link: http://en.wikipedia.org/wiki/Logistic_function

	@param:
		n: given a real number
	@return:
		logistic: return the sigmoid function based on the given number 
	'''

	return 1.0 / (1 + np.exp(-n))
		
def LogisticBatchGD(X, y, alpha=5e-5, threshold=1e-3):
	'''
	Description: This algorithms represents the Logic Batch Gradient Descent algorithm.

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
	hypothese = logistic(np.dot(X, theta))
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

		# hypothese = logistic(theta.T * X)
		hypothese = logistic(np.dot(X, theta))
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f' % (loop, rmse)

		# updating parameters
		theta = theta + alpha * np.dot(xTrans, (y - hypothese))

	return theta