'''
Created on June 22, 2014

@author Rainicy
'''

import numpy as np

from util import RMSE

def LinearSGD(X, y, options):
	'''
	Description: It implements the linear stochastic gradient descent, with 
				regularized and non-regularized. 
				This version is working on the sparse 20 news group. The updating rule is 
				the same, but when dealing with sparse data, it will be little different.

	@params:
		X: training features, list[dict1, dict2, ..., dict_m]
		y: training label, list[label1, label2, ..., label_m]
		options:
			1) numFeatures: number of features, for initialzing theta
			2) alpha: learning rate
			3) threshold: terminated condition
			4) regularized: if using regularization 
			5) lambda: parameter for regularization
	@return:
		theta: weights for the linear model
	'''

	alpha = options['alpha']
	threshold = options['threshold']
	ifRegularized = options['regularized']
	n = options['numFeatures']

	# m: #samples, n:#features
	m = len(y)
	theta = np.zeros(n+1)
	loop = 0
	rmse = 0
	rmse_ = np.inf

	# training processing
	while abs(rmse - rmse_) > threshold:
		loop += 1
		if loop == 1:
			rmse_ = np.inf
		else:
			rmse_ = rmse

		hypothese = calculateHypoes(X, theta)
		rmse = RMSE(hypothese, y)
		print 'Iteration: %d | RMSE: %f'% (loop, rmse)

		# updating parameters
		### 1) Nomal update:
		####	theta = theta + alpha * [(y - hypo) * X]
		### 2) Regularized update:
		#### 	theta_0 = theta_0 + alpha * [(y - hypo) * X]
		####	theta_j = theta_j + alpha * [(y-hypo)*X - lambda*theta_j]. (for j = 1 to n)
		for i in range(m):
			# if i > 1439 and i < 1700:
			h = calHypo(X[i], theta)
			diff = y[i] - h
			theta = updateTheta(X[i], theta, diff, alpha, ifRegularized, options['lambda'])


	return theta


def updateTheta(x, theta, diff, alpha, ifRegularized, lam):
	'''
	Description: Updates the theta by one sample.
	'''
	theta[0] += alpha * diff
	if not ifRegularized:
		for k in x:
			theta[k+1] += alpha * diff * x[k]
	else:
		for k in x:
			theta[k+1] += alpha * (diff * x[k] - lam * theta[k+1])

	return theta

def calculateHypoes(X, theta):
	'''
	Description: Calculates the hypothese for whole samples by given sparse features X, and theta.
	'''
	hypoes = np.zeros(len(X))
	for i in range(len(X)):
		hypoes[i] = calHypo(X[i], theta)
	return hypoes

def calHypo(x, theta):
	'''
	Description: Calculates hypothese just for one sample point

	@params:
		x: dictionary of {index: value}
	'''
	# theta[0] * X[0] = theta[0]
	h = theta[0]
	for k in x:
		h += x[k] * theta[k+1]
	return h




