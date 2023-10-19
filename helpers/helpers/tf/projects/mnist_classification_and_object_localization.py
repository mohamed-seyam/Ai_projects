import tensorflow as tf 
import numpy as np 
import tensorflow_datasets as tfds

from helpers.visualization.plot_metrics import plot_metrics
from helpers.tf.metrics.iou import intersection_over_union

def dataset_to_numpy_util(training_dataset, validation_dataset, N):
    """Pull a batch from the dataset."""

    # get one batch from each: 10000 validation images, N training digits
    batch_training_ds = training_dataset.unbatch().batch(N)

    # eager execution: loop through datasets normally
    if tf.executing_eagerly():
        for validation_digits, (validation_labels, validation_bboxes) in validation_dataset:
            validation_digits = validation_digits.numpy()
            validation_labels = validation_labels.numpy()
            validation_bboxes = validation_bboxes.numpy()
            break
        for training_digits, (training_labels, training_bboxes) in batch_training_ds:
            training_digits = training_digits.numpy()
            training_labels = training_labels.numpy()
            training_bboxes = training_bboxes.numpy()
            break
    # these were hot encoded in the dataset
    validation_labels = np.argmax(validation_labels, axis = 1)
    training_labels = np.argmax(training_labels, axis = 1)

    return (training_digits, training_labels, training_bboxes,
            validation_digits, validation_labels, validation_bboxes)

def detect_hardware():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")
    
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.cluster_spec().as_dict()["worker"])
    
    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print("Running on multiple GPUs:", [gpu.name for gpu in gpus])
    
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print("Running on single GPU:", gpus[0].name)
    
    else:
        strategy = tf.distribute.get_strategy()
        print("Running on CPU")
    
    print("Number of accelerators:", strategy.num_replicas_in_sync)
    return strategy

def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0, 48, dtype = tf.int32) # 48 is 75 - 28 + 1
    ymin = tf.random.uniform((), 0, 48, dtype = tf.int32)
    image = tf.reshape(image, (28,28,1,))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32) / 255.0
 
    xmax = (xmin + 28) / 75
    ymax = (ymin + 28) / 75
    xmin = xmin / 75
    ymin = ymin / 75
    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])


def get_training_dataset(strategy, batch_size = 64):
    with strategy.scope():
        dataset = tfds.load("mnist", split = "train", as_supervised = True, try_gcs = True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(2048, reshuffle_each_iteration = True)
        dataset = dataset.repeat()  # Mandatory for Keras for now
        dataset = dataset.batch(batch_size, drop_remainder = True) # drop_remainder is important on TPU, batch size must be fixed
        dataset = dataset.prefetch(-1) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(batch_size = 64):
    dataset = tfds.load("mnist", split = "test", as_supervised = True, try_gcs = True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()  # Mandatory for Keras for now
    dataset = dataset.batch(batch_size, drop_remainder = True) # drop_remainder is important on TPU, batch size must be fixed
    
    return dataset

def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, kernel_size = 3, activation = 'relu')(inputs)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size = 3, activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    return x

def dense_layers(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    return x

def classifier(inputs):
    classification_output = tf.keras.layers.Dense(10, activation = "softmax", name = "classification")(inputs)
    return classification_output

def bounding_box_regression(inputs):
    bounding_box_regression_output = tf.keras.layers.Dense(4, name = "bounding_box")(inputs)
    return bounding_box_regression_output

def final_model(inputs):
    feature_cnn = feature_extractor(inputs)
    dense_output = dense_layers(feature_cnn)
    classification_output = classifier(dense_output)
    bounding_box_output = bounding_box_regression(dense_output)
    model = tf.keras.Model(inputs = inputs, outputs = [classification_output, bounding_box_output])
    return model

def define_and_compile_model(inputs):
    model = final_model(inputs)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),
                    loss = {"classification": "categorical_crossentropy", 
                            "bounding_box": "mse"},
                    metrics = {"classification": "accuracy", 
                               "bounding_box": "mse"})
    
    return model
    
def main():
    strategy = detect_hardware()
    with strategy.scope():
        training_dataset = get_training_dataset(strategy, BATCH_SIZE)
        validation_dataset = get_validation_dataset(BATCH_SIZE)
    
    (training_digits, training_labels, training_bboxes,
     validation_digits, validation_labels, validation_bboxes) = dataset_to_numpy_util(training_dataset, validation_dataset, 10)

    with strategy.scope():
        inputs = tf.keras.layers.Input(shape = (75,75,1))
        model = define_and_compile_model(inputs)
    
    model.summary()

    epochs = 10
    steps_per_epoch = 60000 / BATCH_SIZE # 60,000 items in this dataset
    validation_steps = 1
    
    history = model.fit(training_dataset, steps_per_epoch = steps_per_epoch, epochs = epochs,
                        validation_data = validation_dataset, validation_steps = validation_steps)
    
    loss, classification_loss, bounding_box_loss, classification_accuracy, bounding_box_mse = model.evaluate(validation_dataset)
    print("Loss: ", loss)

    plot_metrics(history, "classification_accuracy", "Classification Accuracy")
    plot_metrics(history, "bounding_box_mse", "Bounding Box MSE")


    predictions = model.predict(validation_digits, batch_size=64)
    predicted_labels = np.argmax(predictions[0], axis=1)

    predicted_bboxes = predictions[1]

    iou = intersection_over_union(predicted_bboxes, validation_bboxes)

    iou_threshold = 0.6

    print("Number of predictions where iou > threshold(%s): %s" % (iou_threshold, (iou >= iou_threshold).sum()))
    print("Number of predictions where iou < threshold(%s): %s" % (iou_threshold, (iou < iou_threshold).sum()))
        

if __name__ == "__main__":
    BATCH_SIZE = 8
    main()