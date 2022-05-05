"""
Exam
2a) Random classifier

Ossi Koski
"""

import numpy as np
import random


def main():
    X_train, X_test, y_train, y_test = data()

    y_hat = []
    for rand_i in range(len(y_test)):
        y_hat.append(random.randint(0, 1))
    
    acc = accuracy(y_hat, y_test)
    print("Random accuracy: ", acc)

def data():
    X_train = np.array([item.strip().split() for item in open("./data/X_train.dat").readlines()])
    X_test = np.array([item.strip().split() for item in open("./data/X_test.dat").readlines()])

    #X_train = np.loadtxt('./data/X_train.dat', unpack = True)
    #X_train = X_train.reshape(64, 4)
    #X_test = np.loadtxt('./data/X_test.dat', unpack = True)
    #X_test = X_test.reshape(16, 4)

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
