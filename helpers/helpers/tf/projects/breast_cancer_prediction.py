import pandas as pd 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.model_selection import train_test_split

from helpers.visualization.plot_metrics import plot_confusion_matrix
from helpers.tf.metrics.f1_score import F1Score

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def get_dataset(batch_size=32):
    data_file = './data/tf/breast_cancer_data/data.csv'
    col_names = ["id", "clump_thickness", "un_cell_size", "un_cell_shape", "marginal_adheshion", "single_eph_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
    df = pd.read_csv(data_file, names=col_names, header=None)
    
    df.pop("id")
    df = df[df["bare_nuclei"] != '?' ]
    df.bare_nuclei = pd.to_numeric(df.bare_nuclei) 
    df["class"] = df["class"].replace({2: 0, 4: 1})     
    
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    train_stats = train.describe()
    train_stats.pop("class")
    train_stats = train_stats.transpose()
    
    train_y = train.pop("class")
    test_y = test.pop("class")
    
    norm_train_x = norm(train, train_stats)
    norm_test_x = norm(test, train_stats)

    train_dataset = tf.data.Dataset.from_tensor_slices((norm_train_x.values, train_y.values))
    test_dataset = tf.data.Dataset.from_tensor_slices((norm_test_x.values, test_y.values))

    train_dataset = train_dataset.shuffle(len(train)).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset


def base_model(input_shape):
    input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(input)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input, outputs=x)
    return model

def apply_gradient(optimizer, loss_object, model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss_object(y, logits)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return logits, loss_value

def train_data_for_one_epoch(train_dataset, optimizer, loss_object, model, train_acc_metric, train_f1_score, verbose = False):
    losses = []
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        logits, loss_value = apply_gradient(optimizer, loss_object, model, x_batch_train, y_batch_train)
        losses.append(loss_value)
        
        logits = tf.round(logits)
        logits = tf.cast(logits, 'int64')

        train_acc_metric.update_state(y_batch_train, logits)
        train_f1_score.update_state(y_batch_train, logits)

    if verbose:
        print("Training loss for step %s: %.4f" % (int(step), float(loss_value)))

    return losses

def perform_validation(test_dataset, loss_object, model, test_acc_metric, test_f1_score):
    losses = []

    #Iterate through all batches of validation data.
    for x_val, y_val in test_dataset:

        #Calculate validation loss for current batch.
        val_logits = model(x_val) 
        val_loss = loss_object(y_true=y_val, y_pred=val_logits)
        losses.append(val_loss)

        #Round off and cast outputs to either  or 1
        val_logits = tf.cast(tf.round(model(x_val)), 'int64')

        #Update validation metrics
        test_acc_metric.update_state(y_val, val_logits)
        test_f1_score.update_state(y_val, val_logits)
        
    return losses 

def plot_metrics(train_metric, val_metric, metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.plot(train_metric,color='blue',label=metric_name)
    plt.plot(val_metric,color='green',label='val_' + metric_name)
    plt.show()

def main():
    train_dataset, test_dataset = get_dataset(BATCH_SIZE)
    input_shape = train_dataset.element_spec[0].shape
    model = base_model(input_shape)
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    train_f1_score = F1Score()
    test_f1_score = F1Score()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()

    EPOCHS = 5
    epochs_train_losses, epochs_validation_losses = [], []
    for epoch in range(EPOCHS):
        train_losses = train_data_for_one_epoch(train_dataset, optimizer, loss_object, model, train_acc_metric, train_f1_score)
        train_acc = train_acc_metric.result()
        train_f1_score_result = train_f1_score.result()

        validation_losses = perform_validation(test_dataset, loss_object, model, test_acc_metric, test_f1_score)
        validation_acc = test_acc_metric.result()
        validation_f1_score_result = test_f1_score.result()

        epochs_train_losses.append(np.mean(train_losses))
        epochs_validation_losses.append(np.mean(validation_losses))

        print("Epoch: %d, train_loss: %.4f, train_acc: %.4f, train_f1_score: %.4f, val_loss: %.4f, val_acc: %.4f, val_f1_score: %.4f" % 
              (epoch, float(np.mean(train_losses)), float(train_acc), float(train_f1_score_result), float(np.mean(validation_losses)), float(validation_acc), float(validation_f1_score_result)))
        
        train_acc_metric.reset_states()
        test_acc_metric.reset_states()
        train_f1_score.reset_states()
        test_f1_score.reset_states()
    
    plot_metrics(epochs_train_losses, epochs_validation_losses, "Loss", "Loss", ylim=1.0)
    


if __name__ =="__main__":
    BATCH_SIZE = 32
    main()