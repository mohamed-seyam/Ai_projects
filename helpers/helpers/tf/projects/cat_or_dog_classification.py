import tensorflow as tf
import matplotlib.pyplot as plt 

from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from helpers.tf.callbacks.tf_callbacks import TrackAccCallback


def train_val_generators(training_dir:str, 
                         validation_dir:str
                         )->tuple[ImageDataGenerator, ImageDataGenerator]:
    """Creates the training and validation data generators"""

    # Instantiate the ImageDataGenerator class (don't forget to set the arguments to augment the images)
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                        rotation_range=40,
                                        width_shift_range=.2,
                                        height_shift_range=.2,
                                        shear_range=.2,
                                        zoom_range=.2,
                                        horizontal_flip=True,
                                        fill_mode="nearest")

    # Pass in the appropriate arguments to the flow_from_directory method
    train_generator = train_datagen.flow_from_directory(directory=training_dir,
                                                        batch_size=50,
                                                        class_mode="binary",
                                                        target_size=(150, 150))

    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    validation_datagen = ImageDataGenerator(
        rescale = 1/255.
    )

    # Pass in the appropriate arguments to the flow_from_directory method
    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                    batch_size=30,
                                                                    class_mode="binary",
                                                                    target_size=(150, 150))
    return train_generator, validation_generator

def train_cat_or_dog_model(train_gen, validation_gen):
    
    callbacks = TrackAccCallback()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation = "relu", input_shape = (150,150,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(1, activation = "sigmoid")
        
    ])

    # Compile the model
    # Select a loss function compatible with the last layer of your network
    model.compile(loss= tf.keras.losses.BinaryCrossentropy(),
                  optimizer= tf.keras.optimizers.RMSprop(lr = .001),
                  metrics=['accuracy'])     

    print(model.summary())

    # Train the model
    # Your model should achieve the desired accuracy in less than 15 epochs.
    # You can hardcode up to 20 epochs in the function below but the callback should trigger before 15.
    history = model.fit(x=train_gen,
                        validation_data = validation_gen,
                        epochs=10,
                        callbacks=[callbacks]
                       )

    return history

def show_accuracy_curves(history):

    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs=range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.show()
    print("")

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.show()


def main():
    train_dir = "data/tf/dogs_or_cats_data/training"
    validation_dir = "data/tf/dogs_or_cats_data/validation"

    train_gen, validation_gen = train_val_generators(train_dir, validation_dir)
    hist = train_cat_or_dog_model(train_gen, validation_gen) 
    show_accuracy_curves(hist)

if __name__ == "__main__":
    main()