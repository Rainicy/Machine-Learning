'''
Created on June 27, 2014

@author Rainicy
'''

from numpy import *

def loadDigitData():
	'''
	Description: Loading the digit data from 'digit_rec' folder. 

	@return:
		train_x: training features
		train_y: training labels
		test_x:  testing features
		test_y:  testing labels
	'''
	trainFile = '../../data/digit_rec/train_10pc_org.csv'
	testFile = '../../data/digit_rec/test_full_org.csv'
	trainData = genfromtxt(trainFile, delimiter=',')
	testData = genfromtxt(testFile, delimiter=',')

	# get number of features: n
	dump, n = shape(trainData)

	# skip the first row, which is names for each column
	train_x = trainData[1:, :n-1]
	train_y = trainData[1:, -1]
	test_x  = testData[1:, :n-1]
	test_y  = testData[1:, -1]

	return train_x, mat(train_y).T, test_x, mat(test_y).T

def loadDigitTestData():
	'''
	Description: Loading the digit data from 'digit_rec' folder. 

	@return:
		test_x:  testing features
		test_y:  testing labels
	'''
	testFile = '../../data/digit_rec/test_full_org.csv'
	testData = genfromtxt(testFile, delimiter=',')

	# get number of features: n
	dump, n = shape(testData)

	# skip the first row, which is names for each column
	test_x  = testData[1:, :n-1]
	test_y  = testData[1:, -1]

	return test_x, mat(test_y).T

def writeLog(s):
	'''
	Description: Write the log.
	'''
	fileName = 'log'
	with open(fileName, 'a') as file:
		file.write(s + '\n')
		file.close()