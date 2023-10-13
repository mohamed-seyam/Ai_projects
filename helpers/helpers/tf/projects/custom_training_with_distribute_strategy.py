import tensorflow as tf 
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm
import zipfile

from helpers.os.zip import zipdir


class ResNetModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self._feature_extractor = hub.KerasLayer("data/models_data/resnet_50_feature_vector", trainable=False)
        self._classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def __call__(self, inputs):
        x = self._feature_extractor(inputs)
        x = self._classifier(x)
        return x

def get_data():
    splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
    (train_examples, validation_examples, test_examples), info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True, split=splits, data_dir='data/tf/tfds/')
    
    global NUM_EXAMPLES 
    global NUM_CLASSES

    NUM_EXAMPLES = info.splits['train'].num_examples
    NUM_CLASSES = info.features['label'].num_classes

    return train_examples, validation_examples, test_examples

def set_mirror_strategy():
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    return strategy

def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) /255.0
    return image, label 

def set_global_batch_size(batch_size_per_replica, strategy):
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    print('Global batch size: {}'.format(global_batch_size))
    return global_batch_size

def prepare_data_sets(train_exs, validation_exs, test_exs, global_batch_size):
    train_ds = train_exs.shuffle(NUM_EXAMPLES//4).map(format_image).batch(global_batch_size).prefetch(1)
    validation_ds = validation_exs.map(format_image).batch(global_batch_size).prefetch(1)
    test_ds = test_exs.map(format_image).batch(1)
    return train_ds, validation_ds, test_ds

def distribute_datasets(train_ds, validation_ds, test_ds, strategy):
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    validation_dist_ds = strategy.experimental_distribute_dataset(validation_ds)
    test_dist_ds = strategy.experimental_distribute_dataset(test_ds)
    return train_dist_ds, validation_dist_ds, test_dist_ds

def set_train_step_with_dist_strategy(model, optimizer, loss_object, train_accuracy, strategy):
    with strategy.scope():
        def train_step(inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                predictions = model(images)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_accuracy(labels, predictions)
            return loss
        return train_step
        
def set_test_step_with_dist_strategy(model, loss_object, test_loss, test_accuracy, strategy):
    with strategy.scope():
        def test_step(inputs):
            images, labels = inputs
            predictions = model(images)
            loss = loss_object(labels, predictions)
            test_loss(loss)
            test_accuracy(labels, predictions)
            return loss
        return test_step

def set_model_with_dist_strategy(strategy):
    with strategy.scope():
        model = ResNetModel(NUM_CLASSES)
        optimizer = tf.keras.optimizers.Adam()
        checkpoints = tf.train.Checkpoint(optimizer=optimizer, model=model)
        return model, optimizer, checkpoints

def set_loss_object_with_dist_strategy(strategy, global_batch_size):
    with strategy.scope():
        train_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        
        def compute_loss(labels, predictions):
            per_example_loss = train_loss(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        return train_loss, test_loss, compute_loss

def set_accuracy_metric_with_dist_strategy(strategy):
    with strategy.scope():
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy'
        )
        return train_accuracy, test_accuracy
    
def set_distributed_train_step(train_step, strategy):
    with strategy.scope():
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return distributed_train_step

def set_distributed_test_step(test_step, strategy):
    with strategy.scope():
        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))
        return distributed_test_step
    

def main():
    train_exs, validation_exs, test_exs = get_data()
    strategy = set_mirror_strategy()
    global_batch_size = set_global_batch_size(BATCH_SIZE, strategy) 
    train_ds, validation_ds, test_ds = prepare_data_sets(train_exs, validation_exs, test_exs, global_batch_size)
    train_dist_ds, validation_dist_ds, test_dist_ds = distribute_datasets(train_ds, validation_ds, test_ds, strategy)
    model, optimizer, checkpoints = set_model_with_dist_strategy(strategy)
    train_loss, test_loss, compute_loss = set_loss_object_with_dist_strategy(strategy, global_batch_size)
    train_accuracy, test_accuracy = set_accuracy_metric_with_dist_strategy(strategy)
    train_step = set_train_step_with_dist_strategy(model, optimizer, compute_loss, train_accuracy, strategy)
    test_step = set_test_step_with_dist_strategy(model, compute_loss, test_loss, test_accuracy, strategy)
    distributed_train_step = set_distributed_train_step(train_step, strategy)
    distributed_test_step = set_distributed_test_step(test_step, strategy)

    print("Training started...")

    with strategy.scope():
        for epoch in range(EPOCHS):
            # Train Loop 
            total_loss = 0.0
            num_batches = 0

            for x in tqdm(train_dist_ds):
                total_loss += distributed_train_step(x)
                num_batches += 1
            train_loss = total_loss / num_batches

            # Test Loop
            for x in validation_dist_ds:
                distributed_test_step(x)

            template = ("Epoch {}, Loss: {}, Accuracy: {},  Test Loss: {}, Test Accuracy: {}")
            print (template.format(epoch+1, train_loss, train_accuracy.result()*100,  test_loss.result(), test_accuracy.result()*100))

            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
    
    # Save model
    model_save_path = "data/tmp/mirrored_strategy_model/model/"
    tf.saved_model.save(model, model_save_path)

    # zip model
    zipf = zipfile.ZipFile('./mymodel.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(model_save_path, zipf)
    zipf.close()

if __name__ == "__main__":
    NUM_EXAMPLES = None 
    NUM_CLASSES = None 
    EPOCHS = 10 
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 4


    
    main()