import tensorflow as tf 
from helpers.tf.callbacks.tf_callbacks import TrackAccCallback

def fetch_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize data 
    x_train = x_train / 255.0
    
    data_shape = x_train.shape
    print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
    
    return (x_train, y_train), (x_test, y_test)

def train_mnist_using_dense(x_train, y_train):
    
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
    
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    return model, history

def train_mnist_using_convolution(x_train, y_train):
    
    # Instantiate the callback class
    callbacks = TrackAccCallback()
    
    # Define the model
    model = tf.keras.models.Sequential([    
        tf.keras.layers.Conv2D(64, (3,3), activation ="relu", input_shape = (28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation ="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
             
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 512, activation = "relu"),
        tf.keras.layers.Dense(units = 10, activation = "softmax")
        
        
    ]) 
    
    # Compile the model
    model.compile(optimizer='adam',                   
                  loss='sparse_categorical_crossentropy',                   
                  metrics=['accuracy'])     
    
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


    return model, history

def evaluate_model(model, x_test, y_test):
    print("Evaluating the model......")
    model.evaluate(x_test, y_test)

def main():
    (x_train, y_train), (x_test, y_test) = fetch_data()

    model, hist = train_mnist_using_convolution(x_train, y_train)
    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()