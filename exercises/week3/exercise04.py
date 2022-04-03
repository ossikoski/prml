"""
Exercise Set 4: Linear models in machine learning

Ossi Koski
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


def main():
    X = np.loadtxt('X.dat', unpack = True)
    y = np.loadtxt('y.dat', unpack = True)

    # Normalize data
    X[0, :] = (X[0, :] - X[0, :].mean())/X[0, :].std()
    X[1, :] = (X[1, :] - X[1, :].mean())/X[1, :].std()

    # Remap ground truth values
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))
    y = y.flatten()

    # Task 1
    logregr1 = LogisticRegression()
    logregr1.fit(X.T, y)
    y_hat1 = logregr1.predict(X.T)

    # Task 1 with penalty and intercept off
    logregr1b = LogisticRegression(penalty='none', fit_intercept=False)\
        .fit(np.transpose(X), y)
    y_hat1b = logregr1b.predict(np.transpose(X))

    acc_sklearn = accuracy(y_hat1b, y)

    # Task 2
    w = np.array([[1, -1]]).T
    lr = 0.01
    steps = 100
    w_list = np.zeros(shape=(steps, 2, 1))  # For plotting of w trajectory
    acc_list = list()

    for i in range(steps):
        w0 = w[0][0] - lr * gradient_sse(X, y, w, 0)[0]
        w1 = w[1][0] - lr * gradient_sse(X, y, w, 1)[0]

        w = np.array([[w0, w1]]).T
        w_list[i] = w

        y_hat = np.zeros(shape=y.shape)
        for n, _ in enumerate(X.T):
            predn = logsig(np.matmul(w.T, X[:, n]))
            if predn < 0.5:
                y_hat[n] = 0
            if predn >= 0.5:
                y_hat[n] = 1
            
        acc_list.append(accuracy(y_hat, y))

    # Task 3
    w_ml = np.array([[1, -1]]).T
    lr_ml = 0.0008
    steps_ml = 100
    w_list_ml = np.zeros(shape=(steps_ml, 2, 1))  # For plotting of w trajectory
    acc_list_ml = list()

    for i in range(steps_ml):
        w_0 = w_ml[0][0] + lr_ml * gradient_ml(X, y, w_ml, 0)[0]
        w_1 = w_ml[1][0] + lr_ml * gradient_ml(X, y, w_ml, 1)[0]

        w_ml = np.array([[w_0, w_1]]).T
        w_list_ml[i] = w_ml

        y_hat_ml = np.zeros(shape=y.shape)
        for n, _ in enumerate(X.T):
            predn = logsig(np.matmul(w_ml.T, X[:, n]))
            if predn < 0.5:
                y_hat_ml[n] = 0
            if predn >= 0.5:
                y_hat_ml[n] = 1
        
        acc_list_ml.append(accuracy(y_hat_ml, y))

    plot_weights(w_list, w_list_ml, logregr1.coef_)
    plot_acc(acc_list, acc_list_ml, acc_sklearn)

def gradient_sse(X, y, w, i):
    delta = 0
    for n, _ in enumerate(X.T):
        delta += -2 * (y[n] - logsig(np.matmul(w.T, X[:,n]))) * logsig(np.matmul(w.T, X[:,n])) * (1 - logsig(np.matmul(w.T, X[:,n]))) * X[i,n]

    return delta

def gradient_ml(X, y, w, i):
    delta = 0
    for n, _ in enumerate(X.T):
        if y[n] == 0:
            delta += -logsig(np.matmul(-w.T, X[:, n]))*X[i, n]
        if y[n] == 1:
            delta += (1-logsig(np.matmul(-w.T, X[:, n])))*X[i, n]

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

def plot_weights(w_list_sse, w_list_ml, sklearn):
    plt.figure()
    plt.scatter(x=w_list_sse[:,0], y=w_list_sse[:,1], label='SSE')
    plt.scatter(x=w_list_ml[:,0], y=w_list_ml[:,1])
    plt.scatter(x=sklearn[0][0], y=sklearn[0][1])
    plt.show()

def plot_acc(acc_sse, acc_ml, acc_sklearn):
    fig, ax = plt.subplots()
    l = range(len(acc_sse))
    ax.plot(l, acc_sse, label='SSE')
    ax.plot(l, acc_ml, label='ML')
    ax.plot(l, [acc_sklearn]*len(l), label='sklearn')
    leg = ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
