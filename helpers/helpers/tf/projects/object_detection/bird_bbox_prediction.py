import tensorflow as tf
import tensorflow_datasets as tfds


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

def read_image_tfds_with_original_bbox(data):
    """undo the normalization of the bounding box coordinates"""
    image  = data["image"]
    bbox = data["bbox"]

    shape = tf.shape(image)
    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    bbox_list = [bbox[1] * factor_x, 
                 bbox[0] * factor_y, 
                 bbox[3] * factor_x, 
                 bbox[2] * factor_y]
    
    return image, bbox_list

def feature_extractor(inputs):
    mobile_net_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3), include_top = False, weights = "imagenet")
    feature_extractor = mobile_net_model(inputs)
    return feature_extractor

def dense_layer(features):
    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation = "relu")(x)
    x = tf.keras.layers.Dense(512, activation = "relu")(x)
    return x 

def bounding_box_regression(x):
    bounding_box_regression_output = tf.keras.layers.Dense(4, name = "bounding_box")(x)
    return bounding_box_regression_output


def final_model(inputs):
    feature_cnn = feature_extractor(inputs)
    dense_features = dense_layer(feature_cnn)
    bounding_box_output = bounding_box_regression(dense_features)
    model = tf.keras.Model(inputs = inputs, outputs = bounding_box_output)
    return model

def define_and_compile_model():
    inputs = tf.keras.layers.Input(shape = (224, 224, 3))
    model = final_model(inputs)
    model.compile(loss = "mse", optimizer = "adam")
    print(model.summary())
    return model


def get_training_dataset(strategy: tf.distribute.Strategy, batch_size: int):
    with strategy.scope():
        training_dataset, info = tfds.load("caltech_birds2010", split = "train", with_info=True, as_supervised = True, try_gcs = True)
        training_dataset = training_dataset.map(read_image_tfds_with_original_bbox, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        training_dataset = training_dataset.shuffle(512, shuffle_each_iteration = True)
        training_dataset = training_dataset.repeat()
        training_dataset = training_dataset.batch(batch_size = batch_size)
        training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return training_dataset
    
def get_validation_dataset(batch_size: int):
    validation_dataset, info = tf.load("caltech_birds2010", split = "test", with_info = True, as_supervised = True, try_gcs = True)
    validation_dataset = validation_dataset.map(read_image_tfds_with_original_bbox, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size = batch_size)
    validation_dataset = validation_dataset.repeat()
    return validation_dataset



def main():
    strategy = detect_hardware()
    model = define_and_compile_model()
        
    with strategy.scope():
        training_dataset = get_training_dataset(strategy, BATCH_SIZE)
        validation_dataset = get_validation_dataset(BATCH_SIZE)
        length_of_training_dataset = len(training_dataset)
        length_of_validation_dataset = len(validation_dataset)
        print("Length of training dataset:", length_of_training_dataset)
        print("Length of validation dataset:", length_of_validation_dataset)

        steps_per_epoch = length_of_training_dataset // BATCH_SIZE
        if steps_per_epoch % BATCH_SIZE > 0 :
            steps_per_epoch += 1
        
        validation_steps = length_of_validation_dataset // BATCH_SIZE
        if validation_steps % BATCH_SIZE > 0:
            validation_steps += 1
        
        history = model.fit(training_dataset,
                            steps_per_epoch = steps_per_epoch,
                            epochs = EPOCHS,
                            validation_data = validation_dataset,
                            validation_steps = validation_steps)
    
    model.save("bird_bbox_model.h5")

    # retrieve the images only (not bbox) from the validation dataset
 

    


        




if __name__ == "__main__":
    BATCH_SIZE = 8
    EPOCHS = 50
    main()
