'''
Created on Oct. 25, 2014

@author JiaChi Liu, Rainicy
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, inv
from optparse import OptionParser
import operator
from sklearn import metrics
 
def get_data_path():
    parser = OptionParser()
    parser.add_option("--i",dest="input",help="destination of input dataset")
    parser.add_option("--n",dest="numGaussian",help="number of gaussians")
    return parser.parse_args()

def em(data, K, max_iter=300, converged=0.001):
    m, n = data.shape
    # pi = np.array([1.0 / K] * K)
    pi = np.random.rand(K)
    mu = np.random.random_sample((K, n)) * 3
    sigma = [np.identity(n)] * K
    # sigma = [np.random.random_sample((n, n))] * K
    gamma = np.zeros((K, m))
    count = 0
 
    prev_likelihood = 0
    while count < max_iter:
        # E step
        print '==============iteration %s =================' % (count + 1)
        for p in range(m):
            gaussian_vector = gaussians(data[p], n, mu, sigma, pi)
            s = sum(gaussian_vector)
            for k in range(K):
                gamma[k][p] = gaussian_vector[k] / s
        # print 'gamma : %s' % gamma
 
        # M step
        for k in range(K):
            sum_gamma_k = gamma[k].sum()
            # update mu
            mu[k] = np.zeros(n)
            for p in range(m):
                mu[k] += gamma[k][p] * data[p] / sum_gamma_k
            # update sigma
            sigma[k] = np.zeros(sigma[k].shape)
            for p in range(m):
                diff = np.atleast_2d(data[p] - mu[k])
                sigma[k] += (gamma[k][p] * diff.T.dot(diff)) / sum_gamma_k
            # update pi
            pi[k] = sum_gamma_k / m
 
        current_likelihood = max_likelihood(data, gamma, mu, sigma, pi)
        print 'mean : %s' % mu.tolist()
        print 'sigma : %s' % sigma
        print 'pi : %s' % pi.tolist()
        print 'current max likelihood: %s' % current_likelihood
        print ' '
        count += 1
        if abs(current_likelihood - prev_likelihood) <= converged:
            break
        else:
            prev_likelihood = current_likelihood

    return gamma
 
 
def max_likelihood(data, gamma, mu, sigma, pi):
    likelihood = 0
    n = data.shape[1]
    for i in range(len(data)):
        k = np.argmax(gamma[:, i])
        likelihood += np.log(gaussian(data[i], n, mu[k],sigma[k])) + np.log(pi[k])
    return likelihood
 
 
def gaussians(x, n, mu, sigma, pi):
    s = []
    for k in range(len(mu)):
        s.append(pi[k] * gaussian(x, n, mu[k], sigma[k]))
    return s
 
 
def gaussian(x, n, mu, sigma):
    v = 1. / (((2 * np.pi) ** (n / 2)) * (det(sigma) ** 0.5))
    diff = x - mu
    v *= np.exp(-0.5 * diff.T.dot(inv(sigma)).dot(diff))
    return v
 

def predictGamma(gamma):
    m, n = gamma.shape
    predict_y = np.zeros(n)
    for p in range(n):
        inx = np.argmax(gamma[:, p])
        predict_y[p] = inx+1

    return predict_y
 
def plot(X, y, predict_y):
    colors = ['r', 'g', 'b']
    makers = ['o', 'v', 's']
    m, n = X.shape

    for i in range(m):
        x = X[i]
        label = int(y[i])
        label_pred = int(predict_y[i])
        plt.scatter(x[0], x[1], s=60, marker=makers[label-1], c=colors[label_pred-1])
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
    numGaussian = int(opts.numGaussian)

    dataset = np.loadtxt(input_data)
    X = dataset[:, :-1]
    y = dataset[:, -1]

    gamma = em(X, numGaussian)
    predict_y = predictGamma(gamma)
    
    purity = calPurity(y, predict_y, numGaussian)
    print 'Purity is {}'.format(purity)

    NMI = metrics.normalized_mutual_info_score(y, predict_y)
    print 'NMI is {}'.format(NMI)

    plot(X, y, predict_y)
 
if __name__ == '__main__':
    main()

## how to use: 
## python --i 'dataset_path' --n 'number of Gaussian'