import tensorflow as tf 

class QuadraticLayer(tf.keras.layers.Layer):
    def __init__(self, units = 32, activation = None):
        super(QuadraticLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    # Define Layer states
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.a = tf.Variable(name="weights_a", initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)
        self.b = tf.Variable(name="weights_b", initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)
        self.bias = tf.Variable(name = "bias", initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
        super(QuadraticLayer, self).build(input_shape)

    def call(self, inputs):
        squared_input = tf.math.square(inputs)
        squared_input_x_a = tf.matmul(squared_input, self.a)
        input_x_b = tf.matmul(inputs, self.b)

        return self.activation(squared_input_x_a + input_x_b + self.bias)
