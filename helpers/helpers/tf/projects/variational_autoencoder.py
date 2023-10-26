import numpy as np 
import tensorflow as tf 
import time 
from tqdm import tqdm
import matplotlib.pyplot as plt


def preprocess_image(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > .5, 1.0, 0.0).astype("float32")

def batch_and_shuffle_data(images, size, batch_size):
    images = tf.data.Dataset.from_tensor_slices(images).shuffle(size).batch(batch_size)
    return images

def get_mnist_dataset():
    (train_images, _) , (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_image(train_images)
    test_images = preprocess_image(test_images)
    return train_images, test_images


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits


def log_normal_pdf(sample, mean, logvar, raxis = 1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis = raxis)

def compute_loss(model: CVAE, x: tf.Tensor):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = x_logit, labels = x)
    logpx_z = -tf.reduce_sum(cross_entropy, axis = [1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    loss = - tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    return loss


@tf.function
def train_step(x: tf.Tensor, model: CVAE, optimizer: tf.keras.optimizers.Adam):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_and_save_images(model: CVAE, epoch: int, test_sample: tf.Tensor):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
   
   
                


def main():
    train_images, test_images  = get_mnist_dataset() 
    train_dataset = batch_and_shuffle_data(train_images, train_images.shape[0], train_batch_size)
    test_dataset = batch_and_shuffle_data(test_images, test_images.shape[0], test_batch_size)
    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = CVAE(latent_dim)
    

    for epoch in tqdm(range(1, epochs + 1)):
        start_time = time.time()
        for train_x in tqdm(train_dataset):
            train_step(train_x, model, optimizer)
        
        end_time = time.time()
    
        loss = tf.keras.metrics.Mean()
        for test_x in tqdm(test_dataset):
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print("Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {} \n".format(epoch, elbo, end_time - start_time))
        generate_and_save_images(model, epoch-1, test_images)




if __name__ == "__main__":
    latent_dim = 2
    epochs = 10
    train_batch_size = 32
    test_batch_size = 16
    main()