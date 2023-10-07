import csv 
import numpy as np 
import tensorflow as tf

def parse_data_from_input(filename):
  """
  Parses the images and labels from a CSV file
  
  Args:
    filename (string): path to the CSV file
    
  Returns:
    images, labels: tuple of numpy arrays containing the images and labels
  """
  with open(filename) as file:
    csv_reader = csv.reader(file, delimiter=",")
    next(csv_reader, None)
    labels, images = [], []

    for row in csv_reader :
      label = row[0]
      image = row[1:]
      image = np.reshape(image, (28,28))

      labels.append(label)
      images.append(image)

    labels = np.array(labels).astype("float")
    images = np.array(images).astype("float")
    
    return images, labels
  
def train_val_generators(training_images, training_labels, validation_images, validation_labels):
    """Creates the training and validation data generators
    
    Args:
        training_images (array): parsed images from the train CSV file
        training_labels (array): parsed labels from the train CSV file
        validation_images (array): parsed images from the test CSV file
        validation_labels (array): parsed labels from the test CSV file
        
    Returns:
        train_generator, validation_generator - tuple containing the generators
    """

    # In this section you will have to add another dimension to the data
    # So, for example, if your array is (10000, 28, 28)
    # You will need to make it (10000, 28, 28, 1)
    training_images = np.expand_dims(training_images, axis = -1)
    validation_images = np.expand_dims(validation_images, axis = -1)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1/255.
    )

    train_generator = train_datagen.flow(x=training_images,
                                        y=training_labels,
                                        batch_size=32) 

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1/255.
    )

    validation_generator = validation_datagen.flow(x=validation_images,
                                                    y=validation_labels,
                                                    batch_size=32) 


    return train_generator, validation_generator



def create_model():
    """Creates a CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation = "relu", input_shape = (28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)), 
        tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2,2)), 

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(26, activation = "softmax")
    ])

    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = .001),
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=["accuracy"])

    print(model.summary()) 
    
    return model

def main():
    training_file = "data/tf/sign_language_mnist/sign_mnist_train.csv"
    validation_file = "data/tf/sign_language_mnist/sign_mnist_test.csv"
   
    training_images, training_labels = parse_data_from_input(training_file)
    validation_images, validation_labels = parse_data_from_input(validation_file)

    train_gen, validation_gen = train_val_generators(training_images, training_labels, validation_images, validation_labels)

    model = create_model()

    history = model.fit(train_gen,
                    epochs=15,
                    validation_data=validation_gen)


if __name__ == "__main__":
    main()