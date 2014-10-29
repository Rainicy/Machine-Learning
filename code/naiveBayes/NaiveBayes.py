'''
Created on July 24, 2014

@author Rainicy
'''

from numpy import *
from random import *
from scipy.stats import gamma

def train_gamma(X, y):
	'''
	Description: This is trained by the density of gamma-distributions for each features.

	@params:
		X: training features
		y: training y
	@return:
		model:

	'''
	m, n = X.shape

	model = {}
	## calculate prob of spam and nonspam
	p_spam = sum(y==1) * 1.0 / m
	p_nonspam = sum(y==0) * 1.0 / m

	model['p_spam'] = p_spam
	model['p_nonspam'] = p_nonspam

	index_spam = (y==1)
	index_nonspam = (y==0)
	gammas_spam = []
	gammas_nonspam = []

	for i in range(n):
		ga = {}
		x_spam = asarray(X[index_spam, i])
		a, floc, scale = gamma.fit(x_spam)
		ga['a'] = a
		ga['floc'] = floc
		ga['scale'] = scale
		gammas_spam.append(ga)


		ga = {}
		x_nonspam = asarray(X[index_nonspam, i])
		a, floc, scale = gamma.fit(x_nonspam)
		ga['a'] = a
		ga['floc'] = floc
		ga['scale'] = scale
		gammas_nonspam.append(ga)


	model['gammas_spam'] = gammas_spam
	model['gammas_nonspam'] = gammas_nonspam

	return model

def test_gamma(X, model):
	'''
	Description: Use the Naive Bayes model to test on the given missing features testing data.

	@params:
		X: testing features
		model: gamma models, {'p_spam': float, 'p_nonspam': float, 'gammas': list of gamma}
	@return 
		y: predict labels
	'''
	m, n = X.shape

	y = zeros(m)
	prob_spam = ones(m) 
	prob_nonspam = ones(m)
	
	results_spam = zeros((m,n))
	results_nonspam = zeros((m,n))
	for j in range(n):
		gammas = model['gammas_spam'][j]
		results_spam[:,j] = gamma.logpdf(X[:,j], gammas['a'], loc=gammas['floc'], scale=gammas['scale'])
		# print results_spam[:,j]
		# raw_input()
		if inf in results_spam[:, j]:
			print 'inf feature ' + j
		# print prod(results_spam[:,j])
		gammas = model['gammas_nonspam'][j]
		results_nonspam[:, j] = gamma.logpdf(X[:,j], gammas['a'], loc=gammas['floc'], scale=gammas['scale'])
		# print prod(results_nonspam[:,j])
		# raw_input()

	for i in range(m):
		prob_spam[i] = sum(results_spam[i, :]) + log(model['p_spam'])
		prob_nonspam[i] = sum(results_nonspam[i, :]) + log(model['p_nonspam'])
		print prob_spam[i]
		raw_input()
		print prob_nonspam[i]
		raw_input()

	# print prob_spam
	# print prob_nonspam
	# raw_input()
	index_spam = (prob_spam > prob_nonspam)
	index_nonspam = (prob_spam <= prob_nonspam)
	y[index_spam] = 1
	y[index_nonspam] = 0
	return y


def train(X, y):
	'''
	Description: This implements the Naive Bayes Algorithms and using the Lapace Smoothing.
				 This is designed for Spambase two-classes classification. 

	@params:
		X: training features
		y: training labels
	@return:
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
					Phi_y_1[j] += 1
		else:
			for j in range(n):
				if X[i][j] == 1:
					Phi_y_0[j] += 1

	Phi_y_1 = (Phi_y_1 + 1.0) / (num_y_1 + 2.0)
	Phi_y_0 = (Phi_y_0 + 1.0) / (num_y_0 + 2.0)
	Phi_Y = num_y_1 / float(m)

	NB_Model = dict(Phi_y_0=Phi_y_0, Phi_y_1=Phi_y_1, Phi_Y=Phi_Y)
	return NB_Model

def train_missing_value(X, y):
	''' 
	Description: THis implements the Naive Bayes to build the models with randomly missing features.

	@params:
		X: training features
		y: training labels
	@return:
		NB_Model: Naive Bayes model, which includes the parameters:
				  1) Phi_x_1_y_1: size is 1 * #features. For the parameter stored y=1, each feature = 1's prob.
				  2) Phi_x_0_y_1: size is 1 * #features. For the parameter stored y=1, each feature = 0's prob.
				  3) Phi_x_1_y_0: size is 1 * #features. For the parameter stored y=0, each feature = 1's prob.
				  4) Phi_x_0_y_0: size is 1 * #features. For the parameter stored y=0, each feature = 0's prob.
				  5) Phi_Y:	 size is 1 * 1. For the prob of y=1.
	'''
	m, n = X.shape

	## initialize the params
	Phi_x_1_y_0 = zeros(n)
	Phi_x_0_y_0 = zeros(n)
	Phi_x_1_y_1 = zeros(n)
	Phi_x_0_y_1 = zeros(n)
	Phi_Y = 0
	# total number of y=0 and y=1.
	num_y_0 = (y==0).sum()
	num_y_1 = (y==1).sum()

	skip_count = 0
	total_count = 0
	for i in range(m):
		if y[i] == 1:
			for j in range(n):
				if X[i][j] == -1:
					continue
				elif X[i][j] == 1:
					Phi_x_1_y_1[j] += 1
				else:
					Phi_x_0_y_1[j] += 1
		else:
			for j in range(n):
				if X[i][j] == -1:
					continue
				elif X[i][j] == 1:
					Phi_x_1_y_0[j] += 1
				else:
					Phi_x_0_y_0[j] += 1

	Phi_x_1_y_1 = (Phi_x_1_y_1 + 1.0) / (num_y_1 + 2.0)
	Phi_x_0_y_1 = (Phi_x_0_y_1 + 1.0) / (num_y_1 + 2.0)
	Phi_x_1_y_0 = (Phi_x_1_y_0 + 1.0) / (num_y_0 + 2.0)
	Phi_x_0_y_0 = (Phi_x_0_y_0 + 1.0) / (num_y_0 + 2.0)
	Phi_Y = num_y_1 / float(m)

	NB_Model = dict(Phi_x_1_y_1=Phi_x_1_y_1, Phi_x_0_y_1=Phi_x_0_y_1,Phi_x_1_y_0=Phi_x_1_y_0,Phi_x_0_y_0=Phi_x_0_y_0,Phi_Y=Phi_Y)
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

	index_0 = y<0.5
	index_1 = y>=0.5
	y[index_0] = 0
	y[index_1] = 1

	return y

def test_missing_value(X, NB_Model):
	'''
	Description: Use the Naive Bayes model to test on the given missing features testing data.

	@params:
		X: testing features 
		NB_Model: includes:
				  1) Phi_x_1_y_1: size is 1 * #features. For the parameter stored y=1, each feature = 1's prob.
				  2) Phi_x_0_y_1: size is 1 * #features. For the parameter stored y=1, each feature = 0's prob.
				  3) Phi_x_1_y_0: size is 1 * #features. For the parameter stored y=0, each feature = 1's prob.
				  4) Phi_x_0_y_0: size is 1 * #features. For the parameter stored y=0, each feature = 0's prob.
				  5) Phi_Y:	 size is 1 * 1. For the prob of y=1.
	@return: 
		y: predict labels for given X.
	'''
	m, n = X.shape

	y = zeros(m)
	prob_y_1 = ones(m)
	prob_y_0 = ones(m)

	total_count = 0
	skip_count = 0
	for i in range(m):
		for j in range(n):
			if X[i][j] == -1:
				continue
			elif X[i][j] == 1:
				prob_y_1[i] *= NB_Model["Phi_x_1_y_1"][j]
				prob_y_0[i] *= NB_Model["Phi_x_1_y_0"][j]
			else:
				prob_y_1[i] *= NB_Model["Phi_x_0_y_1"][j]
				prob_y_0[i] *= NB_Model["Phi_x_0_y_0"][j]

	y = (prob_y_1 * NB_Model["Phi_Y"]) / (prob_y_1 * NB_Model["Phi_Y"] + prob_y_0 * (1 - NB_Model["Phi_Y"]))

	index_0 = (y<0.5)
	index_1 = (y>=0.5)
	y[index_0] = 0
	y[index_1] = 1

	return y




		
