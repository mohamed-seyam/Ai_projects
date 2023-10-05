import tensorflow as tf 

"""Helper example to understand how to use vars()
# Define a small class MyClass
class MyClass:
    def __init__(self):
        # One class variable 'a' is set to 1
        self.var1 = 1

# Create an object of type MyClass()
my_obj = MyClass()

__dict__ is a Python dictionary that contains the object's instance variables and values as key value pairs.

my_obj.__dict__ returns {'var1': 1}
vars(my_obj) returns {'var1': 1}
vars(my_obj) is equivalent to my_obj.__dict__

# Adding new instance of variable and give it a value 
my_obj.var2 = 2
vars(my_obj)['var3'] = 3 

# vars(my_obj) returns {'var1': 1, 'var2': 2, 'var3': 3}
"""

class Block(tf.keras.Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size = 2, strides = 2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions

        for i in range(repetitions):
            vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides)

    def call(self, inputs):
        x = inputs 
        for i in range(self.repetitions):
            x = vars(self)[f"conv2D_{i}"](x)
        out = self.max_pool(x)
        return out
    
class Vgg(tf.keras.Model):
    def __init__(self, num_classes):
        super(Vgg, self).__init__()
        self.block_a = Block(64, 3, 2)
        self.block_b = Block(128, 3, 2)
        self.block_c = Block(256, 3, 3)
        self.block_d = Block(512, 3, 3)
        self.block_e = Block(512, 3, 3)

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        if num_classes == 2:
            self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')
        else:
            self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    vgg = Vgg(num_classes=10)
    vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    