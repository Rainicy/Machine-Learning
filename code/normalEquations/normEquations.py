'''
Created on May 29, 2014

@author Rainicy
'''

from numpy.linalg import inv
from numpy import transpose
from numpy import dot

def normEquations(X, y):
	'''
	Description: Use Normal Equations Linear Regression to calculate the theta. 
				 theta = (X.T * X).Inverse * X.T * y

	@param:
		X: training features
		y: training labels
	@return:
		theta: the parameters
	'''

	Xtrans = transpose(X)
	# X.T * X
	XTX = dot(Xtrans, X)

	# theta
	theta = dot(dot(inv(XTX), Xtrans), y)

	return theta


