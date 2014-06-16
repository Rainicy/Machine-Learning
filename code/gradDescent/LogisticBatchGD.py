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
		
def LogisticBatchGD(X, y, options):
	'''
	Description: This algorithms represents the Logistic Batch Gradient Descent algorithm.

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
		### 1) Nomal update:
		####	theta = theta + alpha * [(y - hypo) * X]
		### 2) Regularized update:
		#### 	theta_0 = theta_0 + alpha * [(y - hypo) * X]
		####	theta_j = theta_j + alpha * [(y-hypo)*X - lambda*theta_j]. (for j = 1 to n)
		if not ifRegularized:
			theta += alpha * (X.T * (y - hypothese))
		else:
			theta[0] += alpha * (X.T[0] * (y-hypothese))
			theta[1:] += alpha * (X.T[1:] * (y-hypothese) - options['lambda'] * theta[1:])

	return theta