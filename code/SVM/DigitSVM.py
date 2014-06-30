'''
Created on June 29, 2014

@author Rainicy
'''


from numpy import *
import sys
import shelve

from util import loadDigitData, writeLog
import SVM
from SVM import *



def buildSVM(k, l):
	'''
	Description: Build the SVM model for classes in Digit Data.

	@param:
		k: the SVM model for first class, 0<=k<=9
		l: the SVM model for second class, 0<=l<=9

	@procedure
		saves the SVM simplier model between class k and l.
	'''
	## Step 1: load data
	log = "Step 1: loading data..."
	writeLog(log)
	print log
	train_x, train_y, test_x, test_y = loadDigitData()

	# set_printoptions(threshold='nan')

	# extract k, l classes
	K_IndexTrain = nonzero(train_y.A == k)[0]
	L_IndexTrain = nonzero(train_y.A == l)[0]
	IndexTrain = concatenate((K_IndexTrain, L_IndexTrain))
	# random shuffle the array
	IndexTrain = random.permutation(IndexTrain)

	K_IndexTest  = nonzero(test_y.A == k)[0]
	L_IndexTest  = nonzero(test_y.A == l)[0]
	IndexTest  = concatenate((K_IndexTest, L_IndexTest))
	# random shuffle the array
	IndexTest = random.permutation(IndexTest)


	train_x = train_x[IndexTrain]
	train_y = train_y[IndexTrain]
	test_x = test_x[IndexTest]
	test_y = test_y[IndexTest]

	# sets label to -1 and +1
	train_y[train_y==k] = -1
	train_y[train_y==l] = 1
	test_y[test_y==k] = -1
	test_y[test_y==l] = 1

	# scales the features value between [-1~1]
	train_x = train_x/255.0*2 - 1
	test_x = test_x/255.0*2 - 1


	## Step 2: training data
	log = "Step 2: training data..."
	writeLog(log)
	print log

	C = 16
	toler = 0.001
	maxIter = 50
	svmClassifier = SVM.train(train_x, train_y.T, C, toler, maxIter, kernel = ('rbf', 10))
	# saves the model to disk for feature prediction
	svmClassifier.save('./models/svm_' + str(k) + '_' + str(l))
	# simpleSVM = SVMSimpleStruct(svmClassifier)
	# simpleSVM.save('./models/simple_svm_' + str(k_class))

	# # load the model
	# print 'Step 2: loading model...'
	# d = shelve.open('./models/svm_' + str(k) + '_' + str(l))
	# svmClassifier = d['svm']
	# d.close()


	# # Step 3: testing data
	log = "Step 3: testing data..."
	writeLog(log)
	print log
	accuracy = SVM.test(svmClassifier, test_x, test_y)

	## Step 4: show the results 
	log = 'The classify accuracy is: %.3f%%' % (accuracy * 100)  
	print log
	writeLog(log)



def main():
	for i in range(10):
		for j in range(i+1, 10):
			log = '------------{} & {}----------'.format(i, j)
			print log
			writeLog(log)
			buildSVM(i, j)

if __name__ == '__main__':
	main()