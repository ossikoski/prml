"""
Exercise Set 1: Level test
3. Scikit nearest neighbor classiﬁer
Team name in kaggle: Ossi Koski
"""

import csv
import pickle

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    x_tr, y_tr, x_val = data()

    model = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
    print("Starting fit")
    model.fit(x_tr, y_tr)
    print("Starting predict")
    y_pred = model.predict(x_val)
    to_csv(y_pred)
    print("Writing done")

def data():
    with open('./data/training_x.dat', 'rb') as pickleFile:
        x_tr = pickle.load(pickleFile)
    with open('./data/training_y.dat', 'rb') as pickleFile:
        y_tr = pickle.load(pickleFile)
    with open('./data/validation_x.dat', 'rb') as pickleFile:
        x_val = pickle.load(pickleFile)

    num_imgs_tr = len(y_tr)
    num_imgs_val = len(x_val)

    # One row was different
    x_tr[216805] = x_tr[216805][:, :, 0:3]

    x_tr_reshaped = np.asarray(x_tr).reshape(num_imgs_tr, 64, 3)[:, :, 0]
    y_tr_reshaped = np.asarray(y_tr)
    x_val_reshaped = np.asarray(x_val).reshape(num_imgs_val, 64, 3)[:, :, 0]

    return x_tr_reshaped, y_tr_reshaped, x_val_reshaped

def to_csv(y_pred):
    with open('submission01.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id,Class'])
        for i, predicted_class in enumerate(y_pred):
            writer.writerow([i+1, predicted_class])

if __name__ == '__main__':
    main()
