'''
Created on June 10, 2014

@author Rainicy

@ 
'''

from numpy import *
import SVM

## step 1: load data  
print "step 1: load data..."  
dataSet = [] 
labels = []
with open('../../data/testSet.txt', 'r') as file:
	for line in file.readlines():
		line = line.strip().split('\t')
		dataSet.append([float(line[0]), float(line[1])])
		labels.append(float(line[2]))
  
dataSet = mat(dataSet)
labels = mat(labels).T
train_x = dataSet[0:81, :]  
train_y = labels[0:81, :]  
test_x = dataSet[80:101, :]  
test_y = labels[80:101, :]  
  
## step 2: training...  
print "step 2: training..."  
C = 0.6  
toler = 0.001  
maxIter = 50  
svmClassifier = SVM.train(train_x, train_y.T, C, toler, maxIter, kernel = ('rbf', 0))  
  
## step 3: testing  
print "step 3: testing..."  
accuracy = SVM.test(svmClassifier, test_x, test_y)  
  
## step 4: show the result  
print "step 4: show the result..."    
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)  
SVM.show(svmClassifier) 