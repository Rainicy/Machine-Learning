'''
Created on September 17, 2014

@author PanYuan & Rainicy
'''
from numpy import *
import math
from optparse import OptionParser

def get_data_path():
    parser = OptionParser()
    parser.add_option("--i",dest="input",help="destination of input dataset")
    return parser.parse_args()

def entropy(y):
    n = len(y)
    labels = unique(asarray(y))
    value_num = {}
    result = 0
    for label in labels:
        value_num[label] = sum(y[:, 0]==label)

    for num in value_num.values():
        result += (num*1./n) * log2(1. / (num*1./n))
    return result

def majority_value(y):
    labels = unique(asarray(y))
    value_num = {}
    for label in labels:
        value_num[label] = sum(y[:, 0]==label)

    value = float('-inf')
    majority = None
    for label in labels:
        if value_num[label] > value:
            value = value_num[label]
            majority = label
    return majority
    

def choose_split(data):
    m, n = data.shape
    # all_index = array(range(m))

    Min_Entroy = float('inf')
    chosen_feature = None
    chosen_threshold = None
    chosen_left = None
    chosen_right = None

    ## scan features
    for j in range(n-1):
        values = unique(asarray(data[:, j]))
        ## scan samples
        for i in range(len(values)-1):
            ## choose the avarage value as the threshold
            threshold = (values[i] + values[i+1])/2.0
            left_index = where(asarray(data[:, j]) <= threshold)[0]
            right_index = where(asarray(data[:, j]) > threshold)[0]
            # right_index = delete(all_index, left_index, 0)
            ## choose the minimum Entroy as the split feature and threshold
            sum_entroy = (len(left_index)*1./m) * entropy(data[left_index, -1]) + \
                         (len(right_index)*1./m) * entropy(data[right_index, -1])

            if sum_entroy < Min_Entroy:
                Min_Entroy = sum_entroy
                chosen_feature = j
                chosen_threshold = threshold
                chosen_left = left_index
                chosen_right = right_index


    return chosen_feature, chosen_threshold, chosen_left, chosen_right


class Node():
    
    def __init__(self, data, level=0):
        self.data = data
        self.entropy = entropy(data[:, -1])
        self.level = level
        self.threshold = None
        self.feature = None
        self.isLeaf = False
        self.label = None
        self.left_child = None
        self.right_child = None
        

    def split(self,f):
        print 'Current entropy is {}'.format(self.entropy)
        ## stop condition
        if self.entropy <= .3:
            self.isLeaf = True
            self.label = majority_value(self.data[:, -1])
            print 'Build the leaf, and its label is {}'.format(self.label) 
            return

        print 'level: %d' % self.level
        self.feature, self.threshold, left_index, right_index = choose_split(self.data)

        for i in range(self.level):
            f.write("\t")
        f.write("level: %3d entropy: %0.5f " % (self.level, self.entropy))
        f.write("feature: %d threshold: %f sample: %d" % (self.feature,self.threshold,len(self.data)))
        f.write("left: %d right: %d\n" % (len(left_index),len(right_index)))

        self.left_child = Node(self.data[left_index, :], self.level+1)
        self.right_child = Node(self.data[right_index, :], self.level+1)
        self.left_child.split(f)
        self.right_child.split(f)

        return


class Dtree():
    def __init__(self,train_set,test_set,root=None):
        self.root = root
        self.train_set = train_set
        self.test_set = test_set
        
    def make_prediction(self, test_data):
        pointer = self.root
        while (not pointer.isLeaf):
            if (test_data[:, pointer.feature] <= pointer.threshold):
                pointer = pointer.left_child
            else:
                pointer = pointer.right_child
        return pointer.label
            
    def training(self):
        f = open("tree_structure", "w")
        self.root = Node(self.train_set)
        self.root.split(f)
        f.close()
        
    def testing(self, isTesting=True):
        if isTesting:
            data = self.test_set
        else:
            data = self.train_set

        m, n = data.shape
        x = data[:, 0:-1]
        y = data[:, -1]
        acc = 0
        
        for i in range(m):
            predict = self.make_prediction(x[i, :])
            if predict == y[i,0]:
                acc += 1

        return acc*1./m
    

def load(file):
    '''
    Description: Loading the spambase dataset.
    '''
    data = loadtxt(file, delimiter=',')
    testIndex = arange(1, data.shape[0], 10)
    testData = data[testIndex]
    trainData = delete(data, testIndex, 0)

    return mat(trainData), mat(testData)

def main():
    opts,args = get_data_path()
    input_data = opts.input

    trainData, testData = load(input_data)

    tree = Dtree(trainData, testData)
    print 'Start training...'
    tree.training()
    print 'Start testing....'
    trainErr = tree.testing(isTesting=False)
    print 'The Training Accuracy {}'.format(trainErr)

    testErr = tree.testing()
    print 'The Testing Accuracy {}'.format(testErr)
        
    
if __name__=='__main__':
    main()

'''
Input:
python DecisionTree.py --i spambase.data
Output:
The Training Accuracy 0.969572567013
The Testing Accuracy 0.921739130435
'''

