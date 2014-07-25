'''
Created on July 24, 2014

@author Rainicy
'''

from numpy import *

def train(X, y):
	'''
	Description: This implements the Naive Bayes Algorithms and using the Lapace Smoothing.
				 This is designed for Spambase two-classes classification. 

	@params:
		X: training features
		y: training labels
	@return
		NB_Model: Naive Bayes model, which includes the parameters:
				  1) Phi_y_1: size is 1 * #features. For the parameter stored y=1, each feature = 1's prob.
				  2) Phi_y_0: size is 1 * #features. For the parameter stored y=0, each feature = 1's prob.
				  3) Phi_Y:	 size is 1 * 1. For the prob of y=1.
	'''

	m, n = X.shape
	## initialize the params
	Phi_y_0 = zeros(n)
	Phi_y_1 = zeros(n)
	Phi_Y = 0
	# total number of y=0 and y=1.
	num_y_0 = (y==0).sum()
	num_y_1 = (y==1).sum()

	for i in range(m):
		if y[i] == 1:
			for j in range(n):
				if X[i][j] == 1:
					# print X[i][j]
					Phi_y_1[j] += 1
		else:
			for j in range(n):
				if X[i][j] == 1:
					Phi_y_0[j] += 1

	Phi_y_1 = (Phi_y_1 + 1.0) / (num_y_1 + 2.0)
	Phi_y_0 = (Phi_y_0 + 1.0) / (num_y_0 + 2.0)
	Phi_Y = num_y_1 / float(m)

	NB_Model = dict()
	NB_Model["Phi_y_0"] = Phi_y_0
	NB_Model["Phi_y_1"] = Phi_y_1
	NB_Model["Phi_Y"] = Phi_Y
	return NB_Model
		
def test(X, NB_Model):
	'''
	Description: Use the Naive Bayes model to test the given testing data X.

	@params:
		X: testing features 
		NB_Model: includes:
				  1) Phi_y_1: size is 1 * #features. For the parameter stored y=1, each feature = 1's prob.
				  2) Phi_y_0: size is 1 * #features. For the parameter stored y=0, each feature = 1's prob.
				  3) Phi_Y:	 size is 1 * 1. For the prob of y=1.
	@return: 
		y: predict labels for given X.
	'''
	m, n = X.shape

	y = zeros(m)
	prob_y_1 = ones(m)
	prob_y_0 = ones(m)
	for i in range(m):
		for j in range(n):
			if X[i][j] == 1:
				prob_y_1[i] *= NB_Model["Phi_y_1"][j]
				prob_y_0[i] *= NB_Model["Phi_y_0"][j]
			else:
				prob_y_1[i] *= (1 - NB_Model["Phi_y_1"][j])
				prob_y_0[i] *= (1 - NB_Model["Phi_y_0"][j])

	y = (prob_y_1 * NB_Model["Phi_Y"]) / (prob_y_1 * NB_Model["Phi_Y"] + prob_y_0 * (1 - NB_Model["Phi_Y"]))

	y[y<0.5] = 0
	y[y>=0.5] = 1

	return y




		
