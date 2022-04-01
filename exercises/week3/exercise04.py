"""
Exercise Set 3: Linear models in machine learning


Ossi Koski
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import transpose
from sklearn.linear_model import LogisticRegression


def main():
    X = np.loadtxt('X.dat', unpack = True)
    y = np.loadtxt('y.dat', unpack = True)

    # Task 1
    logregr1 = LogisticRegression()
    logregr1.fit(np.transpose(X), y)
    print(logregr1.coef_)
    y_hat1 = logregr1.predict(np.transpose(X))

    print(accuracy(y_hat1, y))

    # Task 1 with penalty and intercept off
    logregr1b = LogisticRegression(penalty='none', fit_intercept=False)\
        .fit(np.transpose(X), y)
    print(logregr1b.coef_)
    y_hat1b = logregr1b.predict(np.transpose(X))

    print(accuracy(y_hat1b, y))

    # Task 2
    w = np.transpose(np.array([1, -1]))
    lr = 0.001

    for i in range(100):
        w_ = w - lr * gradient_sse_own(X, y, w)
        print(w)

def gradient_sse_own(X, y, w):
    """
    d = (2 * np.matmul(np.transpose(y), X) * np.exp( np.matmul(np.transpose(w), X)) )/(1 + np.exp( np.matmul(np.transpose(w), X) ))^2\
        - (2*X*np.exp( np.matmul(np.transpose(w), X) ))/(1 + np.exp( np.matmul(np.transpose(w), X) ))^3
    """
    d = (2 * np.transpose(y) * X) * np.exp( np.transpose(w) * X )/(1 + np.exp( np.transpose(w) * X) )^2\
        - (2*X*np.exp( np.transpose(w) * X) )/(1 + np.exp( np.transpose(w) * X) )^3

def get_data():
    pass

def accuracy(prediction, validation):
    correct = 0
    for i, predicted in enumerate(prediction):
        if predicted == validation[i]:
            correct += 1
            
    return correct/len(prediction)

if __name__ == '__main__':
    main()
