'''
Created on July 24, 2014

@author Rainicy
'''

import util
import NaiveBayes
from numpy import *


def main():

	# trainFile = "../../data/spambase/missing_values/{}_percent_missing_train.txt"
	# testFile = "../../data/spambase/missing_values/{}_percent_missing_test.txt"
	set_printoptions(threshold='nan')
	Accuracy_train = ones(10)
	Accuracy_test = ones(10)
	for i in range(10):
		print "Working on data with {} testing set".format(i)
		# Step 1: loading data
		print "Loading data..."
		# trainX, trainY, testX, testY = util.loadData(trainFile.format(i*10), testFile.format(i*10))
		# data = loadtxt('../../data/spambase/spambase.data', delimiter=',')
		# trainX, trainY, testX, testY = util.initialData(data)
		##### gammas
		data = loadtxt('../../data/spambase/spambase.data', delimiter=',')
		trainX, trainY, testX, testY = util.initialGammaData(data, i)

		# # Step 2: training data
		print "Training data..."
		# model = NaiveBayes.train(trainX, trainY)
		# model = NaiveBayes.train_missing_value(trainX, trainY)
		##### gammas
		model = NaiveBayes.train_gamma(trainX, trainY)


		# # Step 3: predict test data
		print "Predicting data..."
		# predict_y = NaiveBayes.test(testX, model)
		# predict_y = NaiveBayes.test_missing_value(testX, model)
		##### gammas
		# train_y = NaiveBayes.test_gamma(trainX, model)
		test_y = NaiveBayes.test_gamma(testX, model)

		# # # Step 4: Calculate the Accuracy.
		print "Accuracy..."
		# accuracy = sum(predict_y == testY) / float(testY.size)
		# print "Accuracy on testing : {:.2f}%".format(accuracy*100)
		# print "....Done...."
		##### gammas
		Accuracy_train[i] = sum(train_y == trainY) / float(trainY.size)
		print "Accuracy on training : {:.2f}%".format(Accuracy_train[i]*100)
		Accuracy_test[i] = sum(test_y == testY) / float(testY.size)
		print "Accuracy on test : {:.2f}%".format(Accuracy_test[i]*100)

	print "Total average accuracy on training: {:.2f}%".format(mean(Accuracy_train)*100)
	print "Total average accuracy on testing: {:.2f}%".format(mean(Accuracy_test)*100)

if __name__ == '__main__':
	main()