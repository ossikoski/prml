"""
Exercise Set 5: Neural networks

Ossi Koski
"""

from PIL import Image

#import keras
import numpy as np
#import tensorflow as tf
from sklearn.model_selection import train_test_split

def main():
    X_train, X_test, y_train, y_test = get_data()

    print(X_train)

    # Task 2: Define network
    #model = tf.keras.models.Sequential()
    #model.add(keras.layers.Flatten(input_shape=(4096, 3)))
    #model.add(keras.layers.Dense(10, activation='sigmoid'))
    #model.add(keras.layers.Dense(10, activation='sigmoid'))
    #model.add(keras.layers.Dense(10, activation='sigmoid'))

    # Task 3: Compile and train the net
    #loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #model.compile(optimizer='SGD')

def get_data():
    """
    Task 1
    Get picture pixel valeus and split into training and testing data (80-20)
    """
    # Array to save pixel values in
    X = np.zeros(shape=(659, 4096, 3))

    # Ground thruth values
    y = np.ones(shape=(659))
    y[450:] = 2

    i = 0
    for c, r in enumerate([450, 209]):  # Iterate class 1 and 2
        for num_img in range(r):  # Iterate pics
            num_img_str = (3-len(str(num_img)))*'0' + str(num_img)
            img = Image.open(f'./GTSRB_subset_2/class{c+1}/{num_img_str}.jpg', 'r')
            pix_val = list(img.getdata())
            pix_ar = np.array(pix_val)

            X[i] = pix_ar
            i += 1

    return train_test_split(X, y, test_size=0.2)

if __name__ == '__main__':
    main()
