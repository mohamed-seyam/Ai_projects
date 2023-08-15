import numpy as np 
import tensorflow as tf 
"""
A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. 
This will make a 1 bedroom house cost 100k, a 2 bedroom house cost 150k etc.

How would you create a neural network that learns this relationship so that it
would predict a 7 bedroom house as costing close to 400k etc.
"""
def house_model():
    
    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    xs = np.array([1,2,3,4,5,6], dtype = float)
    ys = np.array([1, 1.5, 2, 2.5, 3, 3.5 ], dtype = float)
    
    # Define your model (should be a model with 1 dense layer and 1 unit)
    # Note: you can use `tf.keras` instead of `keras`
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape = [1])])
    
    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer="sgd", loss="mse")
    
    # Train your model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)
    
    ### END CODE HERE
    return model


if __name__ == "__main__":
    # Get your trained model
    model = house_model()
    
    new_x = 7.0
    prediction = model.predict([new_x])[0]
    print(prediction)