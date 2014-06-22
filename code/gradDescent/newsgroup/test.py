'''
Created on June 22, 2014

@author Rainicy
'''

from util import *
from LinearSGD import *

def main():
	'''
	Description: This testing code, will use Linear Stochastic Gradient Descent algorithm to test 
				sparse 20 news groups data. The goal for this is to see the differences between 
				regularized and non-regularized.
				You will see the implements as following:
					1) loading sparse data
					2) Regularized Linear Stochastic Gradient Descent and Nomal Linear SGD
					3) Root Mean Square Error
	'''

	# Part 1: loads data
	# numFeatures means the number of features
	print "Step 1: loading data......"
	trainX, trainY, testX, testY, numFeatures = loadSparseData()

	# Part 2: trains data
	print "Step 2: training ......"
	options = {'numFeatures': numFeatures,'alpha': 5e-7, 'threshold': 1e-3, 'regularized': True, 'lambda': 1000}
	theta = LinearSGD(trainX, trainY, options)

if __name__ == '__main__':
	main()