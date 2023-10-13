import io
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

import imageio

from PIL import Image

from helpers.visualization.display import display_digits

class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs: tf.Tensor, ground_truth:tf.Tensor, display_freq: int, n_samples: int, gif_path: str):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n_samples = n_samples
        self.git_path = gif_path

    def on_epoch_end(self, epoch, logs=None):
        # randomly select the indexes of images
        idx = np.random.choice(len(self.inputs), self.n_samples)
        x_test, y_test = self.inputs[idx], self.ground_truth[idx]
        predictions = np.argmax(self.model.predict(x_test), axis = 1)

        # plot the digits
        display_digits(x_test, predictions, y_test, epoch, n=self.n_samples)

        # Save the figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))

        # Display the digits every 'display_freq' number of epochs
        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        # durations = [1000 // 50] * len(self.images)
        imageio.mimsave(self.git_path, self.images, duration=5)


def test_viscallback():
    # Load example MNIST data and pre-process it
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='linear', input_dim=784))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=64,
          epochs=5,
          verbose=0,
          callbacks=[VisCallback(x_test, y_test, 1, 10, "data/tmp/callbacks/viscallback.gif")])
    
if __name__ == "__main__":
    test_viscallback()  