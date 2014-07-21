'''
Created on June 29, 2014

@author Rainicy
'''


from numpy import *
import sys
import shelve

from util import loadDigitData, loadDigitTestData, writeLog
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
	svmClassifier = SVM.train(train_x, train_y.T, C, toler, maxIter, kernel = ('rbf', 13))
	# saves the model to disk for feature prediction
	svmClassifier.save('./models/svm_' + str(k) + '_' + str(l))
	# simpleSVM = SVMSimpleStruct(svmClassifier)
	# simpleSVM.save('./models/simple_svm_' + str(k_class))

	# # load the model
	# print 'Step 2: loading model..
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


def testSVM():
	## loading data
	log = "Step 1: loading data..."
	writeLog(log)
	print log
	test_x, test_y = loadDigitTestData()
	# scales from -1 to 1
	test_x = test_x/255.0*2 - 1

	# initialize the vote matrix for testing data, Votes[m, 10]
	m, dump = shape(test_y)
	Votes = mat(zeros((m, 10)))

	## testing data
	log = "Step 2: testing data..."
	for i in range(10):
		for j in range(i+1, 10):
			log = "--working on model: " + str(i) + '&' + str(j)
			print log
			writeLog(log)
			# loading the models
			d = shelve.open('./models/svm_' + str(i) + '_' + str(j))
			svmClassifier = d['svm']
			d.close()
			# testing using the given model and votes
			Votes_k, Votes_l = SVM.testDigitScores(svmClassifier, test_x, m)

			# write to the Votes
			Votes[:, i] += Votes_k
			Votes[:, j] += Votes_l

	## saving Votes matrix
	log = "Step 3: saving votes..."
	print log
	writeLog(log)
	d = shelve.open('./models/Votes_Score_noscale')
	d['vote'] = Votes
	d.close()

def predictSVM():
	## loading the Votes
	log = "Step 1: loading votes..."
	writeLog(log)
	print log
	d = shelve.open('./models/Votes_Score_noscale')
	Votes = d['vote']
	d.close()

	## loading the testing data
	log = "Step 2: loading testing data..."
	writeLog(log)
	print log
	test_x, test_y = loadDigitTestData()

	## predict the testing data
	log = "Step 3: predicting the data..."
	writeLog(log)
	print log
	predict_y =  Votes.argmax(axis=1)
	m, n = shape(test_y)
	matchCount = 0
	for i in range(m):
		if predict_y[i] == test_y[i]:
			matchCount += 1
		else:
		# 	print i
		# 	print str(test_y[i]) + '\t' + str(predict_y[i]) + '\t' + str(Votes[i])
		# 	raw_input()
			writeLog(str(test_y[i]) + '\t' + str(predict_y[i]) + '\t' + str(Votes[i]))


	accuracy = float(matchCount) / m
	log =  "step 4: show the result..."
	writeLog(log)
	print log
	log = 'The classify accuracy is: %.3f%%' % (accuracy * 100)  
	writeLog(log)
	print log




def main():
	### building the SVM models
	# for i in range(10):
	# 	for j in range(i+1, 10):
	# 		log = '------------{} & {}----------'.format(i, j)
	# 		print log
	# 		writeLog(log)
	# 		buildSVM(i, j)
	# set_printoptions(threshold='nan')
	# buildSVM(2, 8)

	### build the Votes matrix
	testSVM()

	### predict the results
	predictSVM()

if __name__ == '__main__':
	main()
