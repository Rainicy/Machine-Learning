'''
Created on July 24, 2014

@author Rainicy
'''

import util
import NaiveBayes
from numpy import *


def main():

	trainFile = "../../data/spambase/missing_values/{}_percent_missing_train.txt"
	testFile = "../../data/spambase/missing_values/{}_percent_missing_test.txt"

	for i in range(10):
		print "Working on data with {}% missing values".format(i*10)
		# Step 1: loading data
		# print "Loading data..."
		trainX, trainY, testX, testY = util.loadData(trainFile.format(i*10), testFile.format(i*10))
		# data = loadtxt('../../data/spambase/spambase.data', delimiter=',')
		# trainX, trainY, testX, testY = util.initialData(data)

		# # Step 2: training data
		# print "Training data..."
		# model = NaiveBayes.train(trainX, trainY)
		model = NaiveBayes.train_missing_value(trainX, trainY)

		# # Step 3: predict test data
		# print "Predicting data..."
		# predict_y = NaiveBayes.test(testX, model)
		predict_y = NaiveBayes.test_missing_value(testX, model)

		# # Step 4: Calculate the Accuracy.
		# print "Accuracy..."
		accuracy = sum(predict_y == testY) / float(testY.size)
		print "Accuracy on testing : {:.2f}%".format(accuracy*100)
		print "....Done...."


if __name__ == '__main__':
	main()