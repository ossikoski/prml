"""
Exam
2c) Decision tree classifier

Ossi Koski
"""

import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def main():
    X_train, X_test, y_train, y_test = data()

    # a) Random classifier
    y_hat_a = []
    for rand_i in range(len(y_test)):
        y_hat_a.append(random.randint(0, 1))
    
    acc_a = accuracy(y_hat_a, y_test)
    print('Random accuracy: ', acc_a)

    # b) k-NN classifier
    for k in [1, 2, 3, 4, 5]:
        model_b = KNeighborsClassifier(n_neighbors=k)
        model_b.fit(X_train, y_train)
        y_hat_b = model_b.predict(X_test)
        acc_b = accuracy(y_hat_b, y_test)
        print(f'k-NN accuracy {acc_b} for k={k}')

    # c) Decision tree classifier
    for depth in [1, 2, 3, 4, 5]:
        model_c = DecisionTreeClassifier(max_depth=depth)
        model_c.fit(X_train, y_train)
        y_hat_c = model_c.predict(X_test)
        acc_c = accuracy(y_hat_c, y_test)
        print(f'Decision tree accuracy: {acc_c} for max depth {depth}')

def data():
    X_train = np.loadtxt('./data/X_train.dat', unpack = True)
    X_train = X_train.T
    X_test = np.loadtxt('./data/X_test.dat', unpack = True)
    X_test = X_test.T

    y_train = np.loadtxt('./data/y_train.dat', unpack = True)
    y_test = np.loadtxt('./data/y_test.dat', unpack = True)
    
    return X_train, X_test, y_train, y_test

def accuracy(prediction, validation):
    correct = 0
    for i, predicted in enumerate(prediction):
        if predicted == validation[i]:
            correct += 1

    return correct/len(prediction)
    
if __name__ == '__main__':
    main()
