'''
Created on June 9, 2014

@author: Rainicy

@ref: http://www.manning.com/pharrington/
'''

from numpy import *
import time

def calKernel(X, x, kernel):
	'''
	Description: Calculate the Kernel(K[:, i]) by one column. 

	@param:
		X: X[m, n] entire training features
		x: x[1, n] one sample data, let's say index is i
		kernel: the kernel options {linear | rbf}
	@return:
		K[m, i]: update one column Kernel by given sample data.(Kmi)
	'''
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



class SVMStruct:
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


def calEk(svm, k):
	'''
	Description: Calculate the error for given x(k)
				 Formula: E(k) = f(x(k)) - y(k)
				 		  f(x(k)) = sum(y(i)*alphas(i)*K(i,k)) + b

	@param:
		svm: the SVM struct
		k: the error calculated for sample index k
	@return:
		error: f(x(k)) - y(k) [hypothese(k) - y(k)]
	'''
	hypothese = float(multiply(svm.y, svm.alphas).T * svm.K[:, k] + svm.b)
	error = hypothese - float(svm.y[k])
	return error


def innerLoop(svm, i):
	'''
	Description: Inner loop to update alpha_i and alpha_j.

	@param:
		svm: the SVM struct
		i: the alpha_i index

	@return 
		1: if find alpha_i, which violates KKT conditions, and update alpha_i and alpha_j,
		   otherwise 0.

	'''
	# calculate the Error: Ek = f(x(k))- y(k)
	calEk(svm, i)



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

	startTime = time.time()

	svm = SVMStruct(mat(X), mat(y).T, C, toler, kernel)

	# Termination conditions:
	# 1) Maxmium iteration or 
	# 2) No more alpha need to be changed after go through all the data.
	# 	 which means, all the samples satisfy the KKT conditions.
	iteration = 0
	entireSet = True
	alphaPairChanged = 0
	while (iteration < maxIter) and ((alphaPairChanged > 0) or (entireSet)):
		alphaPairChanged = 0

		# update alphas through whole data set
		if entireSet:
			for i in range(svm.m):
				alphaPairChanged += innerLoop(svm, i)
			print "iter: %d on Entire Set | Alphas Pairs Changed: %d" % (iteration, alphaPairChanged)
		# update alphas through all non-boundary data set
		else:
			nonBoundIndexes = nonzero((svm.alphas > 0) & (svm.alphas < C))[0] # [0] row index
			for i in nonBoundIndexes:
				alphaPairChanged += innerLoop(svm, i)
			print "iter: %d on Non-Bound Set | Alphas Pairs Changed: %d" % (iteration, alphaPairChanged)
		iteration += 1

		# change update order: Entire Set <=> Non-boundary Set
		if entireSet:
			entireSet = False
		elif alphaPairChanged == 0:
			entireSet = True

	# training end
	print 'SVM training costs %fs' % (time.time() - startTime)

	return svm.alphas, svm.b
