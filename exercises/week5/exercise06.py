"""
Exercise Set 5: Convolutional neural networks

Ossi Koski
"""

from PIL import Image

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def main():
    epochs = 20
    X_train, X_test, y_train, y_test = get_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Scale values

    #print("\n\nytrain", y_train)
    #print("\n\nytest", y_test)
    
    # Task 2: Define network
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(10, 3, strides=2, activation='relu', input_shape=(64, 64, 3)))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(10, 3, strides=2, activation='relu', input_shape=(64, 64, 3)))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Task 3: Compile and train the net and compute test set accuracy
    loss_fn = tf.keras.losses.BinaryCrossentropy()  # from_logits=True
    model.compile(optimizer='SGD', loss=loss_fn, metrics=['accuracy'])
    model.summary()

    print("\n\nX_train.shape\n\n", X_train.shape)

    model.fit(X_train, y_train, epochs=epochs)
    
    y_hat = model.predict(X_test)

    for i, pred in enumerate(y_hat):
        if pred > 0.5:
            y_hat[i] = 1
        if pred <= 0.5:
            y_hat[i] = 0

    #print("\n\ny_hat", y_hat)
    print("Accuracy: ", accuracy(y_test, y_hat))

    model.evaluate(X_test,  y_test, verbose=2)


def get_data():
    """
    Get picture pixel values and split into training and testing data (80-20)
    Copied from exercise05
    """
    # Array to save pixel values in
    X = np.zeros(shape=(659, 64, 64, 3))

    # Ground thruth values
    y = np.zeros(shape=(659))
    y[450:] = 1

    i = 0
    for c, r in enumerate([450, 209]):  # Iterate class 1 and 2
        for num_img in range(r):  # Iterate pics
            num_img_str = (3-len(str(num_img)))*'0' + str(num_img)
            img = Image.open(f'../../data/GTSRB_subset_2/class{c+1}/{num_img_str}.jpg', 'r')
            pix_val = list(img.getdata())
            pix_ar = np.array(pix_val).reshape(64, 64, 3)

            X[i] = pix_ar
            i += 1

    return train_test_split(X, y, test_size=0.2, random_state=42)

def accuracy(prediction, validation):
    correct = 0
    for i, predicted in enumerate(prediction):
        if predicted == validation[i]:
            correct += 1

    return correct/len(prediction)

if __name__ == '__main__':
    main()
