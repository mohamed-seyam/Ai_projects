""" - This file go through creating and training a multi-input model.
    - Build siamese network to find the similarity or dissimilarity between items of clothing
    """

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

def euclidean_distance(vects):
    x, y = vects 
    sum_squares = tf.keras.backend.sum(tf.keras.backend.square(x-y), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sum_squares, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def initialize_base_network():
    input = tf.keras.layers.Input(shape=(28,28,), name="base_input")
    x = tf.keras.layers.Flatten(name="flatten_input")(input)
    x = tf.keras.layers.Dense(128, activation='relu', name="first_base_dense")(x)
    x = tf.keras.layers.Dropout(0.1, name="first_dropout")(x)
    x = tf.keras.layers.Dense(128, activation='relu', name="second_base_dense")(x)
    x = tf.keras.layers.Dropout(0.1, name="second_dropout")(x)
    x = tf.keras.layers.Dense(128, activation='relu', name="third_base_dense")(x)

    return tf.keras.Model(inputs=input, outputs=x)


def initialize_siamese_network(base_network, debug):
    left_input = tf.keras.layers.Input(shape=(28,28,), name="left_input")
    right_input = tf.keras.layers.Input(shape=(28,28,), name="right_input")

    # siamese network
    vector_output_a = base_network(left_input)
    vector_output_b = base_network(right_input)

    # measure the similarity of the two vector outputs
    output = tf.keras.layers.Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vector_output_a, vector_output_b])

    model = tf.keras.Model(inputs=[left_input, right_input], outputs=output)

    if debug:
        print(f"{Color.RED} Plotting the siamese network {Color.OFF}")
        tf.keras.utils.plot_model(model, r"data\tmp\network_imgs\siamese_model.png")
        print(f"{Color.RED} plot saved at {Color.OFF}  {Color.GREEN} data\\tmp\\network_imgs\ {Color.OFF} {Color.OFF}")

    return model
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = tf.keras.backend.square(y_pred)
        margin_square = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))
        return tf.keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def plot_metrics(metric_name, title, ylim=5, history=None):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)
    plt.show()

# Matplotlib config
def visualize_images():
    plt.rc('image', cmap='gray_r')
    plt.rc('grid', linewidth=0)
    plt.rc('xtick', top=False, bottom=False, labelsize='large')
    plt.rc('ytick', left=False, right=False, labelsize='large')
    plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
    plt.rc('text', color='a8151a')
    plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts


# utility to display a row of digits with their predictions
def display_images(left, right, predictions, labels, title, n):
    plt.figure(figsize=(17,3))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])
    plt.grid(None)
    left = np.reshape(left, [n, 28, 28])
    left = np.swapaxes(left, 0, 1)
    left = np.reshape(left, [28, 28*n])
    plt.imshow(left)
    plt.figure(figsize=(17,3))
    plt.yticks([])
    plt.xticks([28*x+14 for x in range(n)], predictions)
    for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if predictions[i] > 0.5: t.set_color('red') # bad predictions in red
    plt.grid(None)
    right = np.reshape(right, [n, 28, 28])
    right = np.swapaxes(right, 0, 1)
    right = np.reshape(right, [28, 28*n])
    plt.imshow(right)
    plt.show()


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
        tf.keras.utils.plot_model(base_network, r"data\tmp\network_imgs\siamese_base_model.png")
        print(f"{Color.RED} plot saved at {Color.OFF}  {Color.GREEN} data\\tmp\\network_imgs\ {Color.OFF} {Color.OFF}")

    siamese_model = initialize_siamese_network(base_network, debug)

    rms = tf.keras.optimizers.RMSprop()
    siamese_model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
    history = siamese_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=128,
                        epochs=10,
                        validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]], ts_y))
    
    loss = siamese_model.evaluate(x=[ts_pairs[:,0],ts_pairs[:,1]], y=ts_y)

    y_pred_train = siamese_model.predict([tr_pairs[:,0], tr_pairs[:,1]])
    train_accuracy = compute_accuracy(tr_y, y_pred_train)

    y_pred_test = siamese_model.predict([ts_pairs[:,0], ts_pairs[:,1]])
    test_accuracy = compute_accuracy(ts_y, y_pred_test)

    print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))
    plot_metrics(metric_name='loss', title="Loss", ylim=0.2, history=history)

    y_pred_train = np.squeeze(y_pred_train)
    indexes = np.random.choice(len(y_pred_train), size=10)
    display_images(tr_pairs[:, 0][indexes], tr_pairs[:, 1][indexes], y_pred_train[indexes], tr_y[indexes], "clothes and their dissimilarity", 10)
if __name__ == "__main__":
    main()