"""
ResNet implementation in TensorFlow 2.0.

This module contains the implementation of the ResNet architecture for image classification tasks. 
It includes the IdentityBlock class, which defines a single identity block used in the ResNet architecture, 
and the ReNet class, which defines the full ResNet architecture. 

The ResNet architecture is a deep neural network that uses residual connections to enable training of very deep networks. 
It has been shown to achieve state-of-the-art performance on a variety of image classification tasks.

Example usage:

    model = ReNet(num_classes=10)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

"""

import tensorflow as tf 

class IdentityBlock(tf.keras.Model):
    def __init__(self, filter, kernel_size):
        super(IdentityBlock, self).__init__(name='')
        self.conv1 = tf.keras.layers.Conv2D(filter, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(filter, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.act2 = tf.keras.layers.Activation('relu')

    def call(self, input_tensor):
        x = input_tensor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([x, input_tensor])
        return self.act2(x)
    


class ReNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ReNet, self).__init__(name='')
        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same') 
        self.bn = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPooling2D((3, 3))
        
        # use the identity blocks you just defined
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.max_pool(x)
        x = self.id1a(x)
        x = self.id1b(x)
        x = self.global_pool(x)
        return self.classifier(x)


if __name__ == "__main__":

    
    resnet = ReNet(num_classes=10)
    resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize data 
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape the data to add a channel dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    resnet.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
    