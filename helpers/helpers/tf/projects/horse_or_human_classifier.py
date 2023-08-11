from helpers.tf.callbacks.tf_callbacks import TrackAccCallback
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def fetch_data():
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            './data/tf/horse-or-human-data/training-horse-or-human/',  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=128,
            # Since you used binary_crossentropy loss, you need binary labels
            class_mode='binary')

    # Flow training images in batches of 128 using train_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
            './data/tf/horse-or-human-data/validation-horse-or-human/',  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=32,
            # Since you used binary_crossentropy loss, you need binary labels
            class_mode='binary')

    return train_generator, validation_generator

def train_horse_or_human_model(train_generator, validation_generator):


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
    history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
    
    ### END CODE HERE
    return history

def main():
    train_gen, validation_gen = fetch_data()
    train_horse_or_human_model(train_gen, validation_gen) 

if __name__ == "__main__":
    main()