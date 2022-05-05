"""
Exam
2b) k-NN classifier

Ossi Koski
"""

import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier


def main():
    X_train, X_test, y_train, y_test = data()

    # a) Random classifier
    y_hat = []
    for rand_i in range(len(y_test)):
        y_hat.append(random.randint(0, 1))
    
    acc = accuracy(y_hat, y_test)
    print('Random accuracy: ', acc)

    # b) k-NN classifier
    for k in [1, 2, 3, 4, 5]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_hat_b = model.predict(X_test)
        acc = accuracy(y_hat_b, y_test)
        print(f'k-NN accuracy {acc} for k={k}')

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
