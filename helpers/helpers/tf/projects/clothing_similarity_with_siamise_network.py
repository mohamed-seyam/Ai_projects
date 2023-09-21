""" - This file go through creating and training a multi-input model.
    - Build siamese network to find the similarity or dissimilarity between items of clothing
    """

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random

from colorist import Color

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
            
    return np.array(pairs), np.array(labels)


def create_pairs_on_set(images, labels):
    
    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')
    
    return pairs, y


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def prepare_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # prepare train and test sets
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # normalize values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # create pairs on train and test sets
    tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)
    ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)

    return tr_pairs, tr_y, ts_pairs, ts_y

def initialize_base_network():
    input = tf.keras.layers.Input(shape=(28,28,), name="base_input")
    x = tf.keras.layers.Flatten(name="flatten_input")(input)
    x = tf.keras.layers.Dense(128, activation='relu', name="first_base_dense")(x)
    x = tf.keras.layers.Dropout(0.1, name="first_dropout")(x)
    x = tf.keras.layers.Dense(128, activation='relu', name="second_base_dense")(x)
    x = tf.keras.layers.Dropout(0.1, name="second_dropout")(x)
    x = tf.keras.layers.Dense(128, activation='relu', name="third_base_dense")(x)

    return tf.keras.Model(inputs=input, outputs=x)

def main():
    debug = True
    tr_pairs, tr_y, ts_pairs, ts_y = prepare_data()
    if debug:
        print(f"{Color.GREEN} show images at index 8 {Color.OFF}")
        show_image(ts_pairs[8][0])
        show_image(ts_pairs[8][1])
        print(f"{Color.GREEN} similarity value for index 8 is {ts_y[8]} {Color.OFF}")
        
        print(f"{Color.GREEN} show images at index 0 {Color.OFF}")
        show_image(tr_pairs[:,0][0])
        show_image(tr_pairs[:,0][1])
        print(f"{Color.GREEN} similarity value for index 0 is {ts_y[0]} {Color.OFF}")
        

        print(f"{Color.GREEN} show images at index 1 {Color.OFF}")
        show_image(tr_pairs[:,1][0])
        show_image(tr_pairs[:,1][1])
        print(f"{Color.GREEN} similarity value for index 1 is {ts_y[1]} {Color.OFF}")
    
    base_network = initialize_base_network()
    if debug:
        print(f"{Color.RED} Plotting the base network {Color.OFF}")
        tf.keras.utils.plot_model(base_network, r"data\tmp\network_imgs\siamese_model.png")
        print(f"{Color.RED} plot saved at {Color.OFF}  {Color.GREEN} data\\tmp\\network_imgs\ {Color.OFF} {Color.OFF}")

if __name__ == "__main__":
    main()