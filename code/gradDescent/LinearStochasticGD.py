'''
Created May 27, 2014

@author Rainicy
'''

from numpy import *
from util import RMSE

def LinearStochasticGD(X, y, options):
	'''
	Description: This algorithms represents the Linear Stochastic Gradient Descent algorithm.

	@param:
		X: training features
		y: training labels
		options:	1) alpha: learning rate
					2) threshold: the threshold for terminate the loop
					3) regularized: True if use regularized, otherwise False
					4) lambda: the parameter for regularization
	@return:
		theta: the parameters model
	'''

	alpha = options['alpha']
	threshold = options['threshold']
	ifRegularized = options['regularized']
	
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

		hypothese = X * theta
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f' % (loop, rmse)

		# updating parameters
		### 1) Nomal update:
		####	theta = theta + alpha * [(y - hypo) * X]
		### 2) Regularized update:
		#### 	theta_0 = theta_0 + alpha * [(y - hypo) * X]
		####	theta_j = theta_j + alpha * [(y-hypo)*X - lambda*theta_j]. (for j = 1 to n)
		if not ifRegularized:
			for i in range(0, m):
				h = X[i] * theta
				theta += alpha * (X[i].T * (y[i] - h))
		else:
			for i in range(0, m):
				h = X[i] * theta
				theta[0] += alpha * (X[i].T[0] * (y[i]-h))
				theta[1:] += alpha * (X[i].T[1:] * (y[i]-h) - options['lambda'] * theta[1:])

		# updating theta by each data sample
		# for i in range(0, m):
		# 	h = X[i] * theta
		# 	theta += alpha * (X[i].T * (y[i] - h))

	return theta
