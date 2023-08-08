import tensorflow as tf 
from helpers.tf.callbacks.tf_callbacks import TrackAccCallback

def fetch_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize data 
    x_train = x_train / 255.0
    
    data_shape = x_train.shape
    print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
    
    return (x_train, y_train), (x_test, y_test)

def train_mnist(x_train, y_train):
    
    # Instantiate the callback class
    callbacks = TrackAccCallback()
    
    # Define the model
    model = tf.keras.models.Sequential([         
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 512, activation = "relu"),
        tf.keras.layers.Dense(units = 10, activation = "softmax")
        
        
    ]) 
    
    # Compile the model
    model.compile(optimizer='adam',                   
                  loss='sparse_categorical_crossentropy',                   
                  metrics=['accuracy'])     
    
    # Fit the model for 10 epochs adding the callbacks
    # and save the training history
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


    return history

def main():
    (x_train, y_train), (x_test, y_test) = fetch_data()

    hist = train_mnist(x_train, y_train)


if __name__ == "__main__":
    main()