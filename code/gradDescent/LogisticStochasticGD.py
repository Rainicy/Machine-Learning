'''
Created on May 28, 2014

@author Rainicy
'''

from numpy import *

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

	X = mat(X)
	y = mat(y).T

	# m: #samples, n:#features
	m, n = shape(X)
	# intialize the theta with all 0s
	theta = mat(zeros((n,1)))
	loop = 0

	# initialize the RMSE for terminating the loop
	rmse = 0
	rmse_ = inf

	# stop looping condition
	while abs(rmse - rmse_) > threshold:
		loop += 1
		if loop == 1:
			rmse_ = inf
		else:
			rmse_ = rmse

		hypothese = logistic(X * theta)
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f' % (loop, rmse)

		# updating parameters by each sample
		for i in range(0, m):
			h = logistic(X[i] * theta)
			theta += alpha * (X[i].T * (y[i] - h))

	return theta