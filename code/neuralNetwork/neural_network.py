import numpy as np
import sys


def logistic(n):
    return 1.0 / (1 + np.exp(-n))

def init_data(m, n):
    '''
    Description: initialize input and output dataset.

    @params: m - input data point size
             n - input data features number

    @returns: X - m*n size
              Y - m*n size
    '''
    X = np.eye(m, n)
    Y = np.eye(m, n)

    return X, Y

def neural_network(X, Y, hidden=3, lrate=0.1, max_iters=2000):
    '''
    Description: Given by the input and output and number of hidden units, use the 
                 neural network to calculate the weights. And return it.
    '''
    m, n = X.shape
    # 8 * 3
    weights_1 = np.random.random(size=(n,hidden)) # first layer weights
    # theta_1 
    theta_1 = np.random.random(hidden)
    # 3 * 8
    weights_2 = np.random.random(size=(hidden,n)) # second layer weigts
    theta_2 = np.random.random(n)

    for iterations in range(max_iters):
        sys.stdout.write('\rrunning iterations = {} / {}'.format(iterations, max_iters))
        sys.stdout.flush()
        ## update for each data point.
        for i in range(m):
            x = X[i, :]
            y = Y[i, :]
            ## forward feed
            hypo_hidden = logistic(np.dot(x, weights_1) + theta_1)
            hypo_output = logistic(np.dot(hypo_hidden, weights_2) + theta_2)

            ## Backpropagate, output layer
            Err_output = hypo_output * (1-hypo_output) * (y-hypo_output)
            ## hidden layers
            Err_hidden = hypo_hidden * (1-hypo_hidden) * np.dot(weights_2, Err_output)

            ## update the weights_1 layer and theta_1
            for i in range(n):
                for j in range(hidden):
                    weights_1[i,j] += lrate*x[i]*Err_hidden[j]
                    theta_1[j] += lrate*Err_hidden[j]

            ## update the weights_2 layer and theta_2
            for i in range(hidden):
                for j in range(n):
                    weights_2[i,j] += lrate*hypo_hidden[i]*Err_output[j]
                    theta_2[j] += lrate*Err_output[j]


    return weights_1, theta_1, weights_2, theta_2
    

if __name__ == '__main__':

    m = 8 ## number of data
    n = 8 ## number of features
    X, Y = init_data(m, n)

    weights_1, theta_1, weights_2, theta_2 = neural_network(X, Y, hidden=3, lrate=1, max_iters=3000)

    print '-------------Results------------'
    print 'Input layer \t\t\t\t Hidden layer \t\t\t\t Output Layer'
    for i in range(8):
        x = X[i, :]
        print x,
        hypo_hidden = logistic(np.dot(x, weights_1) + theta_1)
        print hypo_hidden,
        hypo_output = logistic(np.dot(hypo_hidden, weights_2) + theta_2)
        hypo_output[hypo_output>0.8] = 1
        hypo_output[hypo_output<=0.8] = 0
        print hypo_output
