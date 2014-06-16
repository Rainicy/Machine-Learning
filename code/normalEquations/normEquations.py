'''
Created on May 29, 2014

@author Rainicy
'''

from numpy import *

def normEquations(X, y, options):
	'''
	Description: Use Normal Equations Linear Regression to calculate the theta. 
				 theta = (X.T * X).Inverse * X.T * y

	@param:
		X: training features
		y: training labels
		options:	1) regularized: True regularization 
					2) lambda: regularization parameter
	@return:
		theta: the parameters
	'''

	m, n = shape(X)
	X = mat(X)
	y = mat(y).T

	# X.T * X
	XTX = X.T * X

	# update theta
	### 1) Nomal:
	#####	theta = (X.T * X).Inv * X.T * y
	###	2) Regilarization:
	#####	theta = (X.T*X + lambda*eye[]
	if not options['regularized']:
		theta = (XTX.I * X.T) * y
	else:
		diag = eye(n, n)
		diag[0, 0] = 0
		theta = (XTX + options['lambda']*diag).I * X.T * y

	# print options['lambda']
	return theta


