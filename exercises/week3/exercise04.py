"""
Exercise Set 3: Linear models in machine learning


Ossi Koski
"""

import pandas as pd

import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import transpose
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


def main():
    X = np.loadtxt('X.dat', unpack = True)
    y = np.loadtxt('y.dat', unpack = True)

    # Normalize data
    X = (X - X.mean())/np.std(X)

    # For pandas
    data = {'col1': X[0, :], 'col2': X[1, :]}
    df = pd.DataFrame(data=data)

    # Remap ground truth values
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))
    y = y.flatten()

    # Task 1
    logregr1 = LogisticRegression()
    logregr1.fit(X.T, y)
    print(logregr1.coef_)
    y_hat1 = logregr1.predict(X.T)

    print(accuracy(y_hat1, y))

    # Task 1 with penalty and intercept off
    logregr1b = LogisticRegression(penalty='none', fit_intercept=False)\
        .fit(np.transpose(X), y)
    print(logregr1b.coef_)
    y_hat1b = logregr1b.predict(np.transpose(X))

    print(accuracy(y_hat1b, y))

    # Task 2
    w = np.array([[1, -1]]).T
    lr = 0.1
    steps = 1000
    w_list = np.zeros(shape=(steps, 2, 1))  # For plotting of w trajectory

    for i in range(steps):
        w0 = w[0][0] - lr * gradient_sse(X, y, w, 0)[0]
        
        w1 = w[1][0] - lr * gradient_sse(X, y, w, 1)[0]
        w = np.array([[w0, w1]]).T

        w_list[i] = w
        
    print("\nWeights task 2", w)

    plot_weights(w_list, logregr1b.coef_)

    y_hat2 = list()
    for n, _ in enumerate(X):
        predn = logsig(np.matmul(w.T, X[:,n]))
        y_hat2.append(predn)

    print("y_hat2", y_hat2)

    #print(accuracy(y_hat2, y))

    # Task 3


def gradient_sse(X, y, w, i):
    delta = 0
    for n, _ in enumerate(X):
        delta += -2 * (y[n] - logsig(np.matmul(w.T, X[:,n]))) * logsig(np.matmul(w.T, X[:,n])) * (1 - logsig(np.matmul(w.T, X[:,n]))) * X[i,n]

    return delta

def gradient_ml(X, y, w, i):
    delta = 0
    for n, _ in enumerate(X):
        if y[n] == 0:
            delta += -logsig(-w.T*X[n])
        if y[n] == 1:
            delta += (1-logsig(-w.T*X[n]))

    return delta

def logsig(x):
    """
    Sigmoid function
    """
    return 1/(1+np.exp(-x))

def accuracy(prediction, validation):
    correct = 0
    for i, predicted in enumerate(prediction):
        if predicted == validation[i]:
            correct += 1
            
    return correct/len(prediction)

def plot_weights(w_list, sklearn):
    plt.figure()
    plt.scatter(x=w_list[:,0], y=w_list[:,1])
    plt.scatter(x=sklearn[0][0], y=sklearn[0][1])
    plt.show()

if __name__ == '__main__':
    main()
