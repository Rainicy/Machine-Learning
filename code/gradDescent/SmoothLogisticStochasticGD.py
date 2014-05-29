'''
Created on May 28, 2014

@author Rainicy
'''

import numpy as np

from LogisticBatchGD import logistic
from util import RMSE

		
def SmoothLogisticStochasticGD(X, y, alpha=5e-5, threshold=1e-3):
	'''
	Description: This algorithms represents the Smooth Logistic Stochastic Gradient Descent algorithm.
				More details about Smooth Stochastic GD see the link:
				http://blog.csdn.net/zouxy09/article/details/20319673

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

		dataIndex = range(m)
		# updating parameters by each sample
		for i in range(0, m):
			alpha = 0.1 / (1.0 + loop + i) + 0.01
			# random choose the sample 
			randIndex = int(np.random.uniform(0, len(dataIndex)))
			h = logistic(np.sum(X[randIndex] * theta))
			theta = theta + alpha * (y[randIndex] - h) * X[randIndex]
			# remove the sample already used
			del(dataIndex[randIndex])

	return theta