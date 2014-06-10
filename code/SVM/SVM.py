'''
Created on June 9, 2014

@author: Rainicy

@ref: http://www.manning.com/pharrington/
	  http://blog.csdn.net/zouxy09/article/details/17292011
'''

from numpy import *
import matplotlib.pyplot as plt
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
	elif kernel[0] == 'rbf':	# gaussian kernel
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
		self.kernel = kernel
		self.K = mat(zeros((self.m, self.m)))
		for i in range(self.m):	# initialize the kernel matrix
			self.K[:, i] = calKernel(self.X, self.X[i, :], kernel)


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
	hypothese = float(multiply(svm.alphas, svm.y).T * svm.K[:, k] + svm.b)
	error = hypothese - float(svm.y[k])
	return error

def updateEk(svm, k):
	'''
	Description: Update the error cache for alpha_k after optimize alpha_k

	@param:
		svm: SVM struct
		k: the update for sample index k
	'''
	error = calEk(svm, k)
	svm.errorCache[k] = [1, error]

def selectJ(svm, i, Ei):
	'''
	Description: Select alpha_j by Max(E(i) - E(j)).

	@param:
		svm: the SVM struct
		i: the alpha_i index
		Ei: the error by alpha_i
	@return
		j: the alpha_j index
		Ej: the error by alpha_j
	'''

	# we want to choose the alpha_j, which satisfies Max(|E(i) - E(j)|)
	## But we just find alpha_j from non-bound examples in training set
	# set Ei in Error Cache valid
	svm.errorCache[i] = [1, Ei]
	## find all the non-bound candidates for j
	validErrorCacheList = nonzero(svm.errorCache[:,0].A)[0]

	maxDelta = 0
	j = -1
	Ej = 0

	# find the j make the maximum gap
	if len(validErrorCacheList) > 1:
		for k in validErrorCacheList:
			if k == i:
				continue
			else:
				Ek = calEk(svm, k)
				delta = abs(Ei-Ek)
				if delta > maxDelta:
					Ej = Ek
					j = k
					maxDelta = delta
	# the first round random choose j
	else:
		j = i
		while (j==i):
			j = int(random.uniform(0, svm.m))
		Ej = calEk(svm, j)

	return j, Ej

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
	Ei = calEk(svm, i)

	# By given i, let's check if the point violats the KKT conditions
	## First, satisfy KKT conditions:
	###		1) alpha_i = 0 & y_i * f(x_i) >= 1	(outside the boundary)
	###		2) alpha_i = C & y_i * f(x_i) <= 1	(between the boundary)
	###		3) 0<alpha_i<C & y_i * f(x_i) =  1	(on the boundary)
	## Second, violate KKT conditions: 
	####	y(i)E(i) = y(i)(f(x_i) - y(i)) = y(i) * f(x_i) - 1
	####	===> y(i)E(i) >= 0 equal to 1)
	####	===> y(i)E(i) <= 0 equal to 2)
	####	===> y(i)E(i) =  0 equal to 3)
	###		1) y(i)E(i) > 0 & (alpha_i != 0 <==> alpha_i > 0)
	###		2) y(i)E(i) < 0 & (alpha_i != C <==> alpha_i < C)
	###		3) y(i)E(i) = 0 & (it's on the boundary, needless optimized)
	# So just consider the violated conditions 1) & 2)
	if ((svm.y[i] * Ei > svm.toler) and (svm.alphas[i] > 0)) or \
		((svm.y[i] * Ei < -svm.toler) and (svm.alphas[i] < svm.C)):

		# Step 1: select alpha_j

		j, Ej = selectJ(svm, i, Ei)
		alpha_i_old = svm.alphas[i].copy()
		alpha_j_old = svm.alphas[j].copy()

		# Step 2: calculate H & L for j

		# if y(i) != y(j)
		## 1) L = Max(0, alpha_j - alpha_i)
		## 2) H = Min(C, alpha_j - alpha_i + C)
		# if y(i) == y(j)
		## 1) L = Max(0, alpha_i + alpha_j - C)
		## 2) H = Min(C, alpha_i + alpha_j)
		if svm.y[i] != svm.y[j]:
			L = max(0, svm.alphas[j] - svm.alphas[i])
			H = min(svm.C, svm.alphas[j] - svm.alphas[i] + svm.C)
		else:
			L = max(0, svm.alphas[i] + svm.alphas[j] - svm.C)
			H = min(svm.C, svm.alphas[i] + svm.alphas[j])
		# needless optimize
		if L == H:
			return 0

		# Step 3: calculate eta = K(x_i, x_i) + K(x_j, x_j) - 2*K(xi, xj)
		eta = svm.K[i, i] + svm.K[j, j] - 2*svm.K[i,j]
		# eta means the similarity of sample i and j 
		## eta cannot be smaller than 0, it's because:
		## eta = ||phi(x_1) - phi(x_2)||**2
		if eta<=0:
			return 0

		# Step 4: update alpha j
		svm.alphas[j] = alpha_j_old + svm.y[j] * (Ei - Ej) / eta

		# Step 5: clip alpha j
		if svm.alphas[j] < L:
			svm.alphas[j] = L
		elif svm.alphas[j] > H:
			svm.alphas[j] = H

		# update the Ej in errorCache
		updateEk(svm, j)

		# Step 6: if j not moving enough, return 0
		if abs(alpha_j_old - svm.alphas[j]) < 0.00001:
			return 0

		# Step 7: update alpha i
		svm.alphas[i] = alpha_i_old + svm.y[i] * svm.y[j] * (alpha_j_old - svm.alphas[j])

		# Step 8: update b
		bi = svm.b - Ei + (alpha_i_old-svm.alphas[i])*svm.y[i]*svm.K[i,i] + (alpha_j_old-svm.alphas[j])*svm.y[j]*svm.K[j,i]
		bj = svm.b - Ej + (alpha_i_old-svm.alphas[i])*svm.y[i]*svm.K[i,i] + (alpha_j_old-svm.alphas[j])*svm.y[j]*svm.K[j,i]

		if (svm.alphas[i]>0) and (svm.alphas[i]<svm.C):
			svm.b = bi
		elif (svm.alphas[j]>0) and (svm.alphas[j]<svm.C):
			svm.b = bj
		else:
			svm.b = (bi + bj) / 2.0

		# finnally change one pair alphas
		return 1

	else:
		return 0



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
		svm: SVM struct
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
			nonBoundIndexes = nonzero((svm.alphas.A > 0) & (svm.alphas.A < C))[0] # [0] row index
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

	return svm

def test(svm, testX, testY):
	'''
	Description: Test the given test data by the trained svm model.

	@param:
		svm: SVMStruct
		testX: testing features
		testY: testing labels
	@return:
		accuracy: rate of correct predictive labels
	'''
	testX = mat(testX)
	testY = mat(testY)
	[m, n] = shape(testX)
	supportVectorIndex = nonzero(svm.alphas.A > 0)[0]
	supportVectors = svm.X[supportVectorIndex]
	supportVectorsLabels = svm.y[supportVectorIndex]
	supportVectorAlphas = svm.alphas[supportVectorIndex]
	matchCount = 0
	for i in range(m):
		Kernel = calKernel(supportVectors, testX[i, :], svm.kernel)
		hypothese = multiply(supportVectorsLabels, supportVectorAlphas).T * Kernel + svm.b
		if sign(hypothese) == sign(testY[i]):
			matchCount += 1

	accuracy = float(matchCount / m)
	return accuracy

def show(svm):
	'''
	Description: Show trained svm model in 2-D.
	'''
	if shape(svm.X)[1] != 2:
		print 'The dimension must be 2.'
		return 

	# draw training data
	for i in range(svm.m):
		if svm.y[i] == -1:
			plt.plot(svm.X[i, 0], svm.X[i, 1], 'or')
		elif svm.y[i] == 1:
			plt.plot(svm.X[i, 0], svm.X[i, 1], 'ob')

	# mark support vectors
	svIndex = nonzero(svm.alphas.A > 0)[0]
	for i in svIndex:
		plt.plot(svm.X[i, 0], svm.X[i, 1], 'oy')

	# draw the classifer line
	w = zeros((2, 1))
	for i in svIndex:
		w += multiply(svm.alphas[i] * svm.y[i], svm.X[i, :].T)
	min_x = min(svm.X[:, 0])[0, 0]
	max_x = max(svm.X[:, 0])[0, 0]
	y_min_x = float(-svm.b - w[0] * min_x) / w[1]
	y_max_x = float(-svm.b - w[0] * max_x) / w[1]
	plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
	plt.show()

