'''
Created on Oct. 2014

@author Rainicy
'''

from numpy import *

def stumpPredict(X, feature, threshold):
	'''
	Description: predict the label, by given specific column feature.
	'''
	m, n = shape(X)
	predictedY = ones((m, 1))
	predictedY[X[:, feature] <= threshold] = -1.0
	return predictedY

def selectStump(X, y, D, numSteps=10, optimal=True):
	'''
	Description: select stump optimal or randomly.
	'''
	m, n = shape(X)
	selectedStump = {}
	h_t = mat(zeros((m, 1)))
	minError = inf
	## select the optimal stump
	if optimal:
		maxAbsError = -inf
		for i in range(n):
			minValue = X[:, i].min()
			maxValue = X[:, i].max()
			step = (maxValue - minValue) * 1.0 /numSteps
			for j in range(-1, numSteps+1):
				threshold = minValue + j * step
				predictedY = stumpPredict(X, i, threshold)
				error = mat(ones((m, 1)))
				error[predictedY == y] = 0
				weightedError = D.T * error
				absError = abs(0.5 - weightedError)
				if absError > maxAbsError:
					maxAbsError = absError
					minError = weightedError
					h_t = predictedY
					selectedStump["feature"] = i
					selectedStump["threshold"] = threshold
	else:
		feature = random.randint(0,n-1)
		minValue = X[:, feature].min()
		maxValue = X[:, feature].max()
		step = (maxValue - minValue) * 1.0 /numSteps
		threshold = minValue + (random.randint(-1,numSteps+1)) * step
		predictedY = stumpPredict(X, feature, threshold)
		error = mat(ones((m,1)))
		error[predictedY == y] = 0
		weightedError = D.T * error

		minError = weightedError
		h_t = predictedY
		selectedStump["feature"] = feature
		selectedStump["threshold"] = threshold

	return selectedStump, minError, h_t

def trainAdaBoost(X, y, options=None):
	'''
	Description: Adaptive Boosting.
	'''
	maxIterations = options['iter']
	optimal = options['optimal']
	numSteps = options['steps']
	m, n = shape(X)
	weakLearners = []
	# weigh distribution - [m * 1]
	D = mat(ones((m,1))/m)
	# \sum(\alpha_t * h_t(x)) - [m * 1]
	H_t = mat(zeros((m,1)))

	for i in range(maxIterations):
		stump, error, h_t = selectStump(X, y, D, numSteps, optimal)
		print error
		alpha = float(0.5 * log((1.0-error)/error))
		stump['alpha'] = alpha
		weakLearners.append(stump)

		## updating D.
		D = multiply(D, exp(multiply(-1.0 * alpha * y, h_t)))/D.sum()
		H_t += alpha * h_t
		Error = multiply(sign(H_t) != y, ones((m,1)))
		errorRate = Error.sum() * 1.0 / m
		print "Error at round {} : {}".format(i+1, errorRate)
		# raw_input()
		if errorRate == 0.0:
			break

	return weakLearners

def predictAdaBoost(X, weakLearners):
	'''
	Description: predict the y by given X.
	'''
	m, n = shape(X)
	H_t = mat(zeros((m,1)))
	for i in range(len(weakLearners)):
		h_t = stumpPredict(X, weakLearners[i]["feature"],\
				weakLearners[i]["threshold"])
		H_t += weakLearners[i]["alpha"] * h_t
	return sign(H_t)


