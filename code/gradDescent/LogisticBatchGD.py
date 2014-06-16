'''
Created on May 28, 2014

@author Rainicy
'''

from numpy import *

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

	return 1.0 / (1 + exp(-n))
		
def LogisticBatchGD(X, y, alpha=5e-5, threshold=1e-3):
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

	X = mat(X)
	y = mat(y).T

	# m: #samples, n:#features
	m, n = shape(X)
	# intialize the theta with all 0s
	theta = mat(zeros((n,1)))
	loop = 0

	# initialize the RMSE for terminating the loop
	# hypothese = logistic(X*theta)
	rmse = 0
	rmse_ = inf

	# stop looping condition
	while abs(rmse - rmse_) > threshold:
		loop += 1
		if loop == 1:
			rmse_ = inf
		else:
			rmse_ = rmse

		# hypothese = logistic(theta.T * X)
		hypothese = logistic(X * theta)
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f' % (loop, rmse)

		# updating parameters
		theta += alpha * (X.T * (y - hypothese))

	return theta