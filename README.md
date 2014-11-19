# Machine-Learning

This is the study notes on [Machine Learning class(cs229) in Stanford](http://cs229.stanford.edu/). I also refer the materials from the class [CS6140 in Northeastern University](http://www.ccs.neu.edu/home/vip/teach/MLcourse/) to help me implement algorithms.

## Algorithms

#### 1. Supervised Learning
* Naive Bayes
	- [Two-class NB with/without Missing Features](./code/naiveBayes/NaiveBayes.py)	([data: spambase](http://archive.ics.uci.edu/ml/datasets/Spambase))
	- [Multi-class NB with Missing Features](./code/naiveBayes/censusNaiveBayes.py)	([data: censusIncome](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29))

* Linear Regression ([data: spambase](http://archive.ics.uci.edu/ml/datasets/Spambase))
	- [Linear Batch Gradient Descent](./code/gradDescent/LinearBatchGD.py) (Regularized & Nomal)
	- [Linear Stochastic Gradient Descent](./code/gradDescent/LinearStochasticGD.py)
	- [Linear Normal Equations](./code/normalEquations/normEquations.py) (Regularized & Nomal)
	- [Locally Weighted Linear Regression](./code/locallyWeighted/lwr.py) ([Ref: <Machine Learning in Action> Chapter 8](http://www.manning.com/pharrington/))

* Logistic Regression ([data: spambase](http://archive.ics.uci.edu/ml/datasets/Spambase))
	- [Logistic Batch Gradient Descent](./code/gradDescent/LogisticBatchGD.py) (Regularized & Nomal)
	- [Logistic Stochastic Gradient Descent](./code/gradDescent/LogisticStochasticGD.py)
    - [Smooth Stochastic Gradient Descent](./code/gradDescent/SmoothLogisticStochasticGD.py) ([Ref: <Machine Learning in Action>(stocGradAscent1)](https://github.com/pbharrin/machinelearninginaction/blob/master/Ch05/logRegres.py))
    - [Newton-Raphson](./code/newtonRaphson/newton.py)([Ref: Logistic Regression by Jia Li](http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/logit.pdf))

* [Perceptron Algorithm](./code/perceptron/perceptron.py) ([data: spambase](http://archive.ics.uci.edu/ml/datasets/Spambase))

* Support Vector Machine
	- [Sequential minimal optimization(SMO)](./code/SVM/SVM.py)

* Neural Network
	- [Encode/Decode] (./code/neuralNetwork/neural_network.py)

* Classification Tree (data: spambase)
	- [Decision Tree](./code/tree/DecisionTree.py)


#### 2. Unsupervised Learning
* [K-Means](./code/kmeans/kmeans.py)	([toy dataset](./data/toy/))

* [DBSCAN](./code/dbscan/dbscan.py)	([toy dataset](./data/toy/))

* [EM for Mix-Gaussians](./code/em/mixGaussian.py)	([toy dataset](./data/toy/))

#### 3. Study Notes

* Principal Component Analysis (PCA) ----- [Note in [*Latex*](./notes/PCA/) or [*PDF*](http://rainicy.github.io/docs/PCA.pdf)]

* Linear Discriminant Analysis (LDA) ----- [Note in [*Latex*](./notes/LDA/) or [*PDF*](http://rainicy.github.io/docs/LDA.pdf)]

* Support Vector Machine (SVM) ----- [Note in [*Latex*](./notes/SVM/) or [*PDF*](http://rainicy.github.io/docs/svm.pdf)]