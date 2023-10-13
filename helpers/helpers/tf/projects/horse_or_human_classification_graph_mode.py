import tensorflow_datasets as tfds
import tensorflow as tf 
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np


@tf.function
def map_function(img, label):
    image_height = IMAGE_SIZE
    image_width = IMAGE_SIZE
    img = tf.image.resize(img, (image_height, image_width))
    img/= 255.0

    return img, label


def get_data():
    splits, info = tfds.load("horses_or_humans", as_supervised=True, with_info=True, split=["train[:80%]", "train[80%:]", "test"])
    (train_examples, validation_examples, test_examples) = splits
    num_examples = info.splits['train'].num_examples
    global NUM_CLASSES
    NUM_CLASSES = info.features['label'].num_classes
    train_ds  = train_examples.map(map_function).shuffle(num_examples).batch(BATCH_SIZE)
    test_ds = test_examples.map(map_function).batch(BATCH_SIZE)
    validation_ds = validation_examples.map(map_function).batch(BATCH_SIZE)
    return train_ds, validation_ds, test_ds 

def get_model():
    MODULE_HANDLE = "data/models_data/resnet_50_feature_vector"
    model = tf.keras.Sequential([
    hub.KerasLayer(MODULE_HANDLE, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')])
    model.summary()
    return model 


def set_adam_optimizer():
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    return optimizer

def set_sparse_categorical_crossentropy_loss():
    train_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    validation_loss_object= tf.keras.losses.SparseCategoricalCrossentropy()
    return train_loss_object, validation_loss_object

def set_sparse_categorical_accuracy_metric():
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    validation_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    return train_accuracy_metric, validation_accuracy_metric


def train_one_step(model, optimizer, x, y, loss_object, accuracy_metric):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy_metric(y, predictions)
    return loss


def plot_image(i, predictions_array, true_label, img, class_names ):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.squeeze(img)

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    
    # green-colored annotations will mark correct predictions. red otherwise.
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    
    # print the true label first
    print(true_label)
  
    # show the image and overlay the prediction
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
    
@tf.function
def train(model, optimizer, epochs, device, train_ds, train_loss, train_acc, valid_ds, valid_loss, valid_acc):
    step = 0 
    loss = 0.0
    for epoch in range(epochs):
        for x, y in train_ds:
            step += 1 
            with tf.device(device):
                loss = train_one_step(model, optimizer, x, y, train_loss, train_acc)
            
            tf.print('Step', step, 
                    ': train loss', loss, 
                    '; train accuracy', train_acc.result())

        with tf.device(device):
            for x, y in valid_ds:
                y_pred = model(x)
                loss = valid_loss(y, y_pred)
                valid_acc(y, y_pred)
    
    tf.print("validation loss: ", loss , "validation accuracy: ", valid_acc.result())

def evaluate(device, test_ds, model, index):
    test_imgs = []
    test_labels = []

    predictions = []
    with tf.device(device_name=device):
        for images, labels in test_ds:
            preds = model(images)
            preds = preds.numpy()
            predictions.extend(preds)

            test_imgs.extend(images.numpy())
            test_labels.extend(labels.numpy())

    class_names = ['horse', 'human']

    # you can modify the index value here from 0 to 255 to test different images
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(index, predictions, test_labels, test_imgs, class_names)
    plt.show()

def main():
    train_ds, validation_ds, test_ds = get_data()
    model = get_model()
    optimizer = set_adam_optimizer()
    train_loss, validation_loss = set_sparse_categorical_crossentropy_loss()
    train_accuracy, validation_accuracy = set_sparse_categorical_accuracy_metric()
    
    train(model, optimizer, EPOCHS, DEVICE, train_ds, train_loss, train_accuracy, validation_ds, validation_loss, validation_accuracy)

    evaluate(DEVICE, test_ds, model, 8)
    
   

if __name__ == "__main__":
    BATCH_SIZE = 4
    IMAGE_SIZE = 224
    NUM_CLASSES = None
    EPOCHS = 2
    DEVICE = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    main()