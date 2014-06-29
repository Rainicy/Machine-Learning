'''
Created on June 27, 2014

@author Rainicy
'''

from numpy import *

from util import loadDigitData
import SVM
import sys
import shelve


def main(argv):
	# for testing
	set_printoptions(threshold='nan')

	## Step 1: load data
	print "Step 1: loading data..."
	train_x, train_y, test_x, test_y = loadDigitData()

	m = shape(train_y)

	# scalies data from -1 to 1 to work better
	train_x = train_x/255.0*2 - 1
	test_x = test_x/255.0*2 - 1

	# build the k-th class
	k_class = int(argv)
	train_y[train_y!=k_class] = -1
	train_y[train_y==k_class] = 1
	

	# print train_y
	# raw_input()

	## Step 2: training data
	print "Step 2: training data..."
	C = 2.8
	toler = 0.00001
	maxIter = 10000
	svmClassifier = SVM.train(train_x, train_y.T, C, toler, maxIter, kernel = ('rbf', 0.0073))
	svmClassifier.save('./models/svm_' + str(k_class))

	# load the model
	# d = shelve.open('model')
	# svmClassifier = d['svm']


	## Step 3: testing data
	# print "Step 3: testing data..."
	# accuracy = SVM.test(svmClassifier, test_x, test_y)

	# ## Step 4: show the results 
	# print 'The classify accuracy is: %.3f%%' % (accuracy * 100)  

if __name__ == '__main__':
	main(sys.argv[1])
