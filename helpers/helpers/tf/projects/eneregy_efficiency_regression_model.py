import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def format_output(data):
    y1 = data.pop('Y1')
    y1 = np.array(y1)
    y2 = data.pop('Y2')
    y2 = np.array(y2)
    return y1, y2


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def prepare_data(data_path):
    df = pd.read_excel(data_path)
    df = df.sample(frac=1).reset_index(drop=True)

    # Split the data to train test split
    train, test = train_test_split(df, test_size=.2)
    train_stats = train.describe()

    # Get Y1 and Y2 as the 2 outputs and format them as np arrays
    train_stats.pop('Y1')
    train_stats.pop('Y2')
    train_stats = train_stats.transpose()

    train_y = format_output(train)
    test_y = format_output(test)

    norm_train_X = norm(train, train_stats)
    norm_test_X = norm(test, train_stats)

    return norm_train_X, train_y, norm_test_X, test_y

def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(history, metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()


def create_model_with_functional_api(input_shape):

    input_layer = tf.keras.layers.Input(shape = input_shape)
    first_dense = tf.keras.layers.Dense(units = '128', activation = "relu")(input_layer)
    second_dense = tf.keras.layers.Dense(units = '128', activation = "relu")(first_dense)

    # Y1 output 
    y1_output = tf.keras.layers.Dense(units = '1', name = "y1_output")(second_dense)
    third_dense = tf.keras.layers.Dense(units = "64", activation = "relu")(second_dense)

    # Y2 output
    y2_output = tf.keras.layers.Dense(units = '1', name = "y2_output")(third_dense)

    # define model 
    model = tf.keras.Model(inputs = input_layer, outputs = [y1_output, y2_output])

    print(model.summary())

    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        optimizer = optimizer, 
        loss = {"y1_output": "mse", 
                "y2_output" : "mse"},
        metrics = {"y1_output" : tf.keras.metrics.RootMeanSquaredError(), 
                   "y2_output" : tf.keras.metrics.RootMeanSquaredError()}
    )

    return model 

def train_model(model, norm_train_X, train_y, norm_test_X, test_y):
    history = model.fit(norm_train_X, train_y,
                    epochs=500, batch_size=10, validation_data=(norm_test_X, test_y))

    return history

def main():
    data_path = r"data\tf\eneregy_efficiency_data\ENB2012_data.xlsx"
    norm_train_X, train_y, norm_test_X, test_y = prepare_data(data_path)
    
    input_shape = (len(norm_train_X.columns),) 
    model = create_model_with_functional_api(input_shape)
    hist = train_model(model, norm_train_X, train_y, norm_test_X, test_y)

    # Test the model and print loss and mse for both outputs
    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_y)
    print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))

    # Plot the loss and mse
    Y_pred = model.predict(norm_test_X)
    plot_diff(test_y[0], Y_pred[0], title='Y1')
    plot_diff(test_y[1], Y_pred[1], title='Y2')
    plot_metrics(hist, metric_name='y1_output_root_mean_squared_error', title='Y1 RMSE', ylim=6)
    plot_metrics(hist, metric_name='y2_output_root_mean_squared_error', title='Y2 RMSE', ylim=7)

if __name__ == "__main__":
    main()