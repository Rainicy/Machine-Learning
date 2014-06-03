'''
Created May 31, 2014

@author Rainicy
'''

import numpy as np
import matplotlib.pyplot as plt

from util import RMSE, initialData
from lwr import lwr
from regression import *


def main():

	# For testing
	np.set_printoptions(threshold='nan')

	xArr, yArr = loadDataSet('../../data/ex0.txt')

	yHat = np.zeros(len(yArr))
	for i in range(len(yArr)):
		yHat[i] = lwr(xArr, yArr, xArr[i], 0.003)
	# yHat = lwlrTest(xArr, xArr, yArr, 0.003)

	xMat=mat(xArr)
	srtInd = xMat[:,1].argsort(0)
	xSort=xMat[srtInd][:,0,:]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:,1],yHat[srtInd])
	ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0] , s=2,
     c='red')
	plt.show()



if __name__ == '__main__':
	main()