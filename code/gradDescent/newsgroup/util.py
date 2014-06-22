'''
Created on June 22, 2014

@author Rainicy
'''

import re
import numpy as np

def loadSparseData():
	'''
	Description: Loads the sparse 20 news groups data. The data fromat is 
				"label featureIndex:featureValue #some comments"
	@return:
		trainX:	training features
		trainY: training label
		testX: testing features
		testY: testing label
	'''

	# Step 1: read config file to get number of data points
	configTrainFile = '../../../data/newsdata/unigram_tf/train.trec/config.txt'
	configTestFile = '../../../data/newsdata/unigram_tf/test.trec/config.txt'
	with open(configTrainFile, 'r') as file:
		lines = file.readlines()
		numTrainData = int(re.search(r'numDataPoints=(.*)', lines[0]).group(1))
		numFeatures = int(re.search(r'numFeatures=(.*)', lines[1]).group(1))
		file.close()
	with open(configTestFile, 'r') as file:
		lines = file.readlines()
		numTestData = int(re.search(r'numDataPoints=(.*)', lines[0]).group(1))
		numFeatures = max(numFeatures, int(re.search(r'numFeatures=(.*)', lines[1]).group(1)))
		file.close()

	# Step 2: initialize the data
	trainX = []
	trainY = []
	testX = []
	testY = []

	# Step 3: read training data and testing data
	trainFile = '../../../data/newsdata/unigram_tf/train.trec/feature_matrix.txt'
	testFile = '../../../data/newsdata/unigram_tf/test.trec/feature_matrix.txt'
	# pattern for "featureIndex:featureValue"
	p = re.compile(r'\d+:\d+')
	r = re.compile(r'(.*):(.*)')
	# training
	with open(trainFile, 'r') as file:
		lines = file.readlines()
		for line in lines:
			dict_x = {}
			y = int(line.split(' ')[0])
			X = p.findall(line)
			ifSkip = False
			for x in X:
				# 'index:value' 
				indexValue = re.search(r, x)
				index = int(indexValue.group(1))
				value = int(indexValue.group(2))
				dict_x[index] = value
				if value > 50:
					ifSkip = True


			# append the data to the results
			if ifSkip:
				continue
			else:
				trainX.append(dict_x)
				trainY.append(y)

	# testing
	with open(testFile, 'r') as file:
		lines = file.readlines()
		for line in lines:
			dict_x = {}
			y = int(line.split(' ')[0])
			X = p.findall(line)
			ifSkip = False
			for x in X:
				# 'index:value' 
				indexValue = re.search(r, x)
				index = int(indexValue.group(1))
				value = int(indexValue.group(2))
				dict_x[index] = value
				if value > 50:
					ifSkip = True

			# append the data to the results
			if ifSkip:
				continue
			else:
				testX.append(dict_x)
				testY.append(y)

	# Step 4: return results
	return trainX, trainY, testX, testY, numFeatures


def RMSE(h, y):
	'''
	Description: Root Mean Squared Error(RMSE). J = sum(h - y)^2
				RMSE = sqrt(J/m). [m: #samples]
				Find more info on: 
				http://en.wikipedia.org/wiki/Root_mean_square_deviation

	@param:
		h: hypothese, calculated by (h = theta.T * X)
		y: the true label
	@return:
		RMSE: root mean Squared error
	'''
	J = 0
	for i in range(len(y)):
		J += (h[i] - y[i])**2
	return np.sqrt(J/len(y))
