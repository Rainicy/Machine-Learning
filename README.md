# Machine-Learning

This is the study notes on [Machine Learning class(cs229) in Stanford](http://cs229.stanford.edu/). I also refer the materials from the class [CS6140 in Northeastern University](http://www.ccs.neu.edu/home/vip/teach/MLcourse/) to help me implement algorithms.

## Algorithms

#### 1. Supervised Learning
* Naive Bayes
	- [Two-class NB with/without Missing Features](./code/naiveBayes/NaiveBayes.py)([data: spambase](http://archive.ics.uci.edu/ml/datasets/Spambase))
	- [More-class NB with Missing Features](./code/naiveBayes/censusNaiveBayes.py)([data: censusIncome](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29))

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


#### 2. Unsupervised Learning
* Principal Component Analysis (PCA)    See in [*Latex*](./notes/PCA/) or [*PDF*](http://rainicy.github.io/docs/PCA.pdf)
* Linear Discriminant Analysis (LDA)    See in [*Latex*](./notes/LDA/) or [*PDF*](http://rainicy.github.io/docs/LDA.pdf)