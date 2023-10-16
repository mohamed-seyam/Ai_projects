"""This a simple model to perform transfer learning on the CIFAR-10 dataset using ResNet50.
"""
import tensorflow as tf
from helpers.visualization.plot_metrics import plot_metrics


def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(32, 32, 3),
                                                              include_top=False,
                                                              weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

def cifar_model(inputs):
    resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
    feature_cnn = feature_extractor(resize)
    classification_output = classifier(feature_cnn)
    return classification_output

def get_model():
    inputs = tf.keras.layers.Input(shape = (32, 32, 3))
    classification_output = cifar_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0001),
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])
    return model

def get_dataset():
    (training_images, training_labels), (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
    training_images = training_images.astype('float32')
    training_images = tf.keras.applications.resnet50.preprocess_input(training_images)
    validation_images = validation_images.astype('float32')
    validation_images = tf.keras.applications.resnet50.preprocess_input(validation_images)
    return training_images, training_labels, validation_images, validation_labels


def main():
    training_images, training_labels, validation_images, validation_labels = get_dataset()
    model = get_model()
    model.summary()
    history = model.fit(training_images, training_labels, epochs = 1, validation_data = (validation_images, validation_labels))
    loss, accuracy = model.evaluate(validation_images, validation_labels)
    plot_metrics(history, 'loss', "Loss")
    plot_metrics(history, 'accuracy', "Accuracy")

if __name__ == "__main__":
    main()