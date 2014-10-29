'''
Created on Oct 25, 2014

@author Rainicy
'''

from numpy import *
from random import shuffle
import operator
from optparse import OptionParser
import matplotlib.pyplot as plt
from sklearn import metrics

NOISE = -2
UNVISITED = -1
VISITED = 0

def get_data_path():
    parser = OptionParser()
    parser.add_option("--i",dest="input",help="destination of input dataset")
    parser.add_option("--e",dest="eps",help="EPS for DBSCAN")
    parser.add_option("--n",dest="minPoint",help="minimum points")
    return parser.parse_args()

def epsNeighbor(p, q, Eps):
	'''
	Description: check if the p, q points is smaller than Eps or not.
	'''
	distance = sqrt(sum((p-q)**2))
	return distance < Eps

def findNeighbors(X, Eps, p_index):
	'''
	Description: returns the given p' neighbors indexes, which is smaller than the eps.
	'''
	m, n = X.shape
	neighbors = []
	for i in range(m):
		if i != p_index:
			if epsNeighbor(X[p_index, :],X[i, :], Eps):
				neighbors.append(i)
	return neighbors

def DBSCAN(X, Eps, MinPts):
	'''
	Description: Density Based Spatial Clustering pf Applications with Noise. 

	@ref: http://www.ccs.neu.edu/home/yzsun/classes/2014Fall_CS6220/Slides/04Matrix_Data_Clustering_1.pdf
	
	@params: 
		1) X: training features.
		2) Eps: maximum radius of the neighborhood
		3) MinPts: minimum number of points in an Eps-neighborhood of that point.
	@returns:
		returns a clusters array, indicate each x point belongs to which cluster. Or Noise. 
	'''

	m, n = X.shape
	## initialize the nodes
	labels = [UNVISITED] * m

	## shuffle the nodes order
	shuffledOrder = range(m)
	shuffle(shuffledOrder)

	# starting cluster
	cur_cluster = 0

	## Algorithm begins
	# randomly choose index
	for p_index in shuffledOrder:
		p = X[p_index, :]
		if labels[p_index] == UNVISITED:
			labels[p_index] = VISITED
			epsNeighborsP = findNeighbors(X, Eps, p_index)
			## if the e-neighbors of p is smaller than MinPts.
			if (len(epsNeighborsP) < MinPts):
				labels[p_index] = NOISE
			else:
				# create a new cluster C, and add p to C
				cur_cluster += 1
				labels[p_index] = cur_cluster
				while (len(epsNeighborsP) > 0):
					q_index = epsNeighborsP[0]
					if labels[q_index] == UNVISITED:
						labels[q_index] = VISITED
						epsNeighborsQ = findNeighbors(X, Eps, q_index)
						if (len(epsNeighborsQ) >= MinPts):
							epsNeighborsP += epsNeighborsQ
					if labels[q_index] <= VISITED:
						labels[q_index] = cur_cluster

					epsNeighborsP = epsNeighborsP[1:]

	return asarray(labels)

def calPurity(y, predict_y):
	'''
	Description: calculate purity.
	'''
	K = len(unique(y))
	n = y.shape[0]

	sumPurity = 0
	for j in range(1, K+1):
		print j
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

def plot(X, y, predict_y):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    makers = ['o', 'v', 's']
    m, n = X.shape

    for i in range(m):
        x = X[i]
        label = int(y[i])
        label_pred = int(predict_y[i])
        if (label_pred == NOISE):
        	plt.scatter(x[0], x[1], s=60, marker=makers[label-1], c='k')
        elif (label_pred > len(colors)):
        	plt.scatter(x[0], x[1], s=60, marker=makers[label-1], c='k')
        else:
        	plt.scatter(x[0], x[1], s=60, marker=makers[label-1], c=colors[label_pred-1])
    plt.show()

def main():
    opts,args = get_data_path()
    input_data = opts.input
    eps = float(opts.eps)
    minPoint = int(opts.minPoint)

    dataset = loadtxt(input_data)
    X = dataset[:, :-1]
    y = dataset[:, -1]

    predict_y = DBSCAN(X, eps, minPoint)
    print predict_y

    purity = calPurity(y, predict_y)
    print 'Purity is {}'.format(purity)

    NMI = metrics.normalized_mutual_info_score(y, predict_y)
    print 'NMI is {}'.format(NMI)

    plot(X, y, predict_y)

if __name__ == '__main__':
	main()
		








