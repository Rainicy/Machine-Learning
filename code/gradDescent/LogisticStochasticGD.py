'''
Created on May 28, 2014

@author Rainicy
'''

import numpy as np

from LogisticBatchGD import logistic
from util import RMSE

		
def LogisticStochasticGD(X, y, alpha=5e-5, threshold=1e-3):
	'''
	Description: This algorithms represents the Logistic Stochastic Gradient Descent algorithm.

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

		hypothese = logistic(theta.T * X)
		hypothese = logistic(np.dot(X, theta))
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f' % (loop, rmse)

		# updating parameters by each sample
		for i in range(0, m):
			h = logistic(np.sum(X[i] * theta))
			theta = theta + alpha * (y[i] - h) * X[i]

	return theta