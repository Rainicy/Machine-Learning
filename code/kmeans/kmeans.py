'''
Created on Oct 24, 2014
@author Rainicy
'''
from numpy import *
import operator
from optparse import OptionParser
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer

def get_data_path():
    parser = OptionParser()
    parser.add_option("--i",dest="input",help="destination of input dataset")
    parser.add_option("--k",dest="K",help="K nearest neighbors")
    parser.add_option("--t",dest="threshold",help="stop threshold")
    return parser.parse_args()

def cluster(X, u, K):
	m, n = shape(X)
	C = zeros(m)
	for i in range(m):
		x = X[i, :]
		c = -1
		minDistance = float('inf')
		for j in range(K):
			distance = sum((x-u[j])**2)
			if (distance < minDistance):
				minDistance = distance
				c = j
		C[i] = c+1
	return C

def calDist(X, u, C):

	m, n = shape(X)
	dist = 0.0
	for i in range(m):
		x = X[i, :]
		c = u[C[i]-1]
		d = sum((x-c)**2)
		dist += d
	return dist



def kmeans(X, K, threshold):
	'''
	Description: The KNN algorithms. 
	@params:
		1) X: training features.
		2) K: k nereast neighbors.
		3) threshold: convergent threshold
	@returns:
		the array of cluster centroids.
	'''
	m, n = shape(X)

	## initiliaze the cluster centroids.
	u = random.random_sample((K, n))

	pre_dist = float('inf')
	iteration = 0
	while iteration < threshold:
		iteration += 1
		## assign the cluster for each x
		C = cluster(X, u, K)

		## update cluster centroids
		for j in range(1, K+1):
			j_index = (C == j)
			u[j-1] = mean(X[j_index], axis=0)

		## check the distotion function
		cur_dist = calDist(X, u, C)
		print 'Current iterations {} and distance is {}'.format(iteration, cur_dist)
		# if abs(cur_dist - pre_dist) < threshold:
		# 	break
		# else:
		# 	pre_dist = cur_dist
	return u


def plot(X, y, predict_y):
    colors = ['r', 'g', 'b']
    makers = ['o', 'v', 's']
    m, n = X.shape

    for i in range(m):
        x = X[i]
        label = int(y[i])
        label_pred = int(predict_y[i])
        plt.scatter(x[0], x[1], marker=makers[label-1], c=colors[label_pred-1])
    plt.show()

def calPurity(y, predict_y, K):
	'''
	Description: calculate purity.
	'''
	n = y.shape[0]

	sumPurity = 0
	for j in range(1, K+1):
		j_index = (predict_y == j)
		true_labels = y[j_index]
		dict_C = {}
		# print len(true_labels)
		for i in range(len(true_labels)):
			if true_labels[i] not in dict_C:
				dict_C[true_labels[i]] = 1
			else:
				dict_C[true_labels[i]] += 1

		sortedCounts = sorted(dict_C.iteritems(), key=operator.itemgetter(1), reverse=True)
		sumPurity += sortedCounts[0][1]

	return sumPurity * 1.0 / n
	sumPurity

def main():
    opts,args = get_data_path()
    input_data = opts.input
    K = int(opts.K)
    threshold = float(opts.threshold)

    dataset = loadtxt(input_data)
    X = dataset[:, :-1]
    y = dataset[:, -1]

    u = kmeans(X, K, threshold)

    predict_y = cluster(X, u, K)

    purity = calPurity(y, predict_y, K)
    print 'Purity is {}'.format(purity)

    NMI = metrics.normalized_mutual_info_score(y, predict_y)
    print 'NMI is {}'.format(NMI)

    plot(X, y, predict_y)

if __name__ == '__main__':
	# main()
	main1()