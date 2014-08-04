'''
Created on August 3, 2014

@author Rainicy
'''

from numpy import *
import copy
import csv


def load(fileName):
	'''
	Description: Load the data by given file path. Returns features and labels dictionary arrays.
	'''

	data = []
	## transform the continuous features to smaller or bigger two situations.
	# continuous_list = [0,2,3,5,16,17,18,24,39]
	continuous_list = [5, 16, 17, 18, 24] 
	for i in range(42):
		data.append(dict())
		if i in continuous_list:
			data[i]["sum"] = 0.0
			data[i]["count"] = 0
	# print len(data)

	numSamples = 0
	with open(fileName, 'r') as csvFile:
		reader = csv.reader(csvFile, delimiter=',')
		for row in reader:
			numSamples += 1
			for i in range(len(row)):
				if i in continuous_list:
					data[i]["sum"] += float(row[i])
					data[i]["count"] += 1
					continue
				if row[i] not in data[i]:	# insert new key
					data[i][row[i]] = 0

	for i in continuous_list:
		data[i]["avg"] = data[i]["sum"]/data[i]["count"]
		data[i]["smaller"] = 0
		data[i]["bigger"] = 0

	# for i in range(len(data)):
	# 	print str(i),
	# 	print len(data[i]),
	# 	print data[i].keys()
	# 	if "avg" in data[i]:
	# 		print data[i]["avg"]
	# 	raw_input()
	return data, numSamples

def train(fileName, data): 
	'''
	Description: Given by the dictionary arrays for the data, and now use the dictionary to compute 
				the parameters. E.X. Phi
	'''
	# continuous_list = [0,2,3,5,16,17,18,24,39]
	continuous_list = [5, 16, 17, 18, 24]  
	pos_label = ' 50000+.'
	neg_label = ' - 50000.'
	pos_data = copy.deepcopy(data)	# " 50000+."
	neg_data = copy.deepcopy(data)	# " - 50000."
	with open(fileName, 'r') as csvFile:
		reader = csv.reader(csvFile, delimiter=',')
		for row in reader:
			if row[-1] == pos_label:	## positive label
				for i in range(len(row)):
					if '?' in row[i]:
						continue
					elif i in continuous_list:
						if float(row[i]) <= pos_data[i]["avg"]:
							pos_data[i]["smaller"] += 1
						else:
							pos_data[i]["bigger"] += 1
					else:
						pos_data[i][row[i]] += 1

			else:
				for i in range(len(row)):
					if '?' in row[i]:
						continue
					elif i in continuous_list:
						if float(row[i]) <= neg_data[i]["avg"]:
							neg_data[i]["smaller"] += 1
						else:
							neg_data[i]["bigger"] += 1
					else:
						neg_data[i][row[i]] += 1

	for i in range(len(pos_data)-1):
		for k in pos_data[i].keys():
			pos_data[i][k] = (float(pos_data[i][k])+1)/(pos_data[-1][pos_label]+2)
			neg_data[i][k] = (float(neg_data[i][k])+1)/(neg_data[-1][neg_label]+2)

	total = pos_data[-1][pos_label] + neg_data[-1][neg_label]
	pos_data[-1][pos_label] = float(pos_data[-1][pos_label])/total
	neg_data[-1][neg_label] = float(neg_data[-1][neg_label])/total
	return pos_data, neg_data

def test(fileName, pos_data, neg_data, m):
	'''
	Description: By given testing data features, and test file name, predict the labels.
	'''
	# continuous_list = [0,2,3,5,16,17,18,24,39]
	continuous_list = [5, 16, 17, 18, 24] 
	pos_label = ' 50000+.'
	neg_label = ' - 50000.'

	y = zeros(m)
	prob_y_pos = ones(m)
	prob_y_neg = ones(m)
	count = -1

	with open(fileName, 'r') as csvFile:
		reader = csv.reader(csvFile, delimiter=',')
		for row in reader:
			count += 1
			for i in range(len(row)-1):
				if '?' in row[i]:
					continue
				elif i in continuous_list:
					if float(row[i]) <= pos_data[i]["avg"]:
						prob_y_pos[count] *= pos_data[i]["smaller"]
						prob_y_neg[count] *= neg_data[i]["smaller"]
					else:
						prob_y_pos[count] *= pos_data[i]["bigger"]
						prob_y_neg[count] *= neg_data[i]["bigger"]
				else:
					prob_y_pos[count] *= pos_data[i][row[i]]
					prob_y_neg[count] *= neg_data[i][row[i]]

	y = (prob_y_pos * pos_data[-1][pos_label]) / (prob_y_pos * pos_data[-1][pos_label] + prob_y_neg * neg_data[-1][neg_label])

	pos_index = (y>=0.5)
	neg_index = (y<0.5)
	y[pos_index] = 1
	y[neg_index] = 0

	total = -1
	match = 0
	with open(fileName, 'r') as csvFile:
		reader = csv.reader(csvFile, delimiter=',')
		for row in reader:
			total += 1
			if (y[total] == 1) and (row[-1] == pos_label):
				match += 1
			elif (y[total] == 0) and (row[-1]==neg_label):
				match += 1

	accuracy = match / float(m)
	print "Accuracy on testing : {:.2f}%".format(accuracy*100)




def main():
	trainFile = '../../data/census-income/census-income.data'
	testFile = '../../data/census-income/census-income.test'

	print "loading ..."
	trainSet, dump = load(trainFile)
	print "done ......"
	print "training ..."
	pos_data, neg_data = train(trainFile, trainSet)
	print "done ......"
	print "testing ..."
	testSet, m = load(testFile)
	test(testFile, pos_data, neg_data, m)
	print "done ......"

if __name__ == '__main__':
	main()

