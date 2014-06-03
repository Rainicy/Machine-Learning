'''
Created on June, 3 2014

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
		
def newton(X, y, threshold=1e-4):
	'''
	Description: This algorithm represents the Logistic Newton-Raphson Descent algorithm.

	@param:
		X: training features
		y: training labels
		threshold: the threshold for terminate the loop
	@return:
		theta: the parameters model
	'''

	# m: #samples, n:#features
	m, n = np.shape(X)
	# intialize the theta with all 0s
	theta = np.zeros(n)
	loop = 0

	# transform arr to matrix and initialize RMSE
	X = np.mat(X)
	Y = np.mat(y).T
	theta = np.mat(theta).T

	# initialize the RMSE for terminating the loop
	# hypothese = logistic(X * theta)
	rmse = 0
	rmse_ = np.inf

	# stop looping condition
	while abs(rmse - rmse_) > threshold:
	# while loop < maxIter:
		loop += 1
		if loop == 1:
			rmse_ = np.inf
		else:
			rmse_ = rmse

		hypothese = logistic(X * theta)
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f' % (loop, rmse)

		# updating parameters
		# convert matrix to array
		arr_hypo = np.squeeze(np.asarray(hypothese))
		innerProduct = arr_hypo * (1 - arr_hypo)
		innerProduct = np.diag(np.asarray(innerProduct))

		# calculate (-X.T*W*X).inv
		xTWx = - X.T * innerProduct * X
		xTWxInv = np.linalg.inv(xTWx)
		# X.T * (y - h)
		xY = X.T * (Y - hypothese)

		# update theta
		theta = theta - xTWxInv * xY

	return theta