from helpers.tf.callbacks.tf_callbacks import TrackAccCallback
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def fetch_data():
    gen = ImageDataGenerator(rescale = 1/255.)

    train_generator = gen.flow_from_directory(
        directory = "./data/tf/happy_or_sad_data",
        target_size = (150, 150),
        batch_size = 10,
        class_mode = "binary"
    )

    return train_generator

def train_happy_or_sad_model(gen):
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
    history = model.fit(x=gen,
                        epochs=20,
                        callbacks=[callbacks]
                       )
    
    ### END CODE HERE
    return history

def main():
    gen = fetch_data()
    train_happy_or_sad_model(gen) 

if __name__ == "__main__":
    main()