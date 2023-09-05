from tensorflow.keras.preprocessing.image import ImageDataGenerator


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