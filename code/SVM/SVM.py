'''
Created on June 9, 2014

@author: Rainicy

@ref: http://www.manning.com/pharrington/
'''

from numpy import *

def calKernel(X, x, kernel):
	m = shape(X)[0]
	K = mat(zeros((m, 1)))
	if kernel[0] == 'linear':	# linear kernel
		K = X * x.T 	# K[m, 1] = X[m, n] * x.T[n, 1]
	else if kernel[0] == 'rbf':	# gaussian kernel
		sigma = kernel[1]
		if sigma == 0:
			sigma = 1
		for i in range(m):
			diff = X[i, :] - x
			K[i] = exp(diff * diff.T / (-2.0 * sigma**2))
	else:
		raise NameError('The Kernel by given is not recognized.')

	return K



class optStruct:
	'''
	Description: class for saving SVM model data
	'''
	def __init__(self, X, y, C, toler, kernel):	# initialize the struct by given params
		self.X = X
		self.y = y
		self.C = C
		self.toler = toler
		self.m = shape(X)[0]
		self.alphas = mat(zeros((self.m, 1)))
		self.b = 0
		self.errorCache = mat(zeros((self.m, 2)))
		self.K = mat(zeros((self.m, self.m)))
		for i in range(self.m):	# initialize the kernel matrix
			self.K[:, i] = calKernel(self.X, self.x[i, :], kernel)



def train(X, y, C, toler, maxIter, kernel):
	'''
	Description: Training entrance.

	@param:
		X: 		training features 
		y: 		training labels
		C:		slack variable
		toler:	termination condition for iteration
		maxIter: maxmium iteration
		kernel: kernel type {linear | rbf}

	@return
		alphas: refer to notes
		b: refer to notes
	'''

	svm = optStruct(mat(X), mat(y).T, C, toler, kernel)

	# Training loop until:
	# 1) Maxmium iteration
	# 2) No more alpha need to be changed after go through all the data.
	# 	 which means, all the samples satisfy the KKT conditions.
	iteration = 0
