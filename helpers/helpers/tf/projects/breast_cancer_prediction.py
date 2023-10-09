import pandas as pd 
import tensorflow as tf 

from sklearn.model_selection import train_test_split

from helpers.visualization.plot_metrics import plot_confusion_matrix


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def get_dataset(batch_size=32):
    data_file = './data/data.csv'
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


def main():
    train_dataset, test_dataset = get_dataset(BATCH_SIZE)
    model = base_model(len(train_dataset.element_spec[0][0]))
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    outputs = model()

if __name__ =="__main__":
    BATCH_SIZE = 32
    main()