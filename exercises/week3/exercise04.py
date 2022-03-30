"""
Exercise Set 3: Linear models in machine learning


Ossi Koski
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


def main():
    X = np.loadtxt('X.dat', unpack = True)
    y = np.loadtxt('y.dat', unpack = True)

    # Task 1
    logregr1 = LogisticRegression(max_iter=2, warm_start=True)
    for i in range(100):
        logregr1.fit(np.transpose(X), y)
        print(logregr1.coef_)
    y_hat1 = logregr1.predict(np.transpose(X))

    print(accuracy(y_hat1, y))

    logregr1b = LogisticRegression(penalty='none', fit_intercept=False)\
        .fit(np.transpose(X), y)
    print(logregr1b.coef_)
    y_hat1b = logregr1b.predict(np.transpose(X))

    print(accuracy(y_hat1b, y))

    # Task 2

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
