import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from helpers.ai.io import get_train_generator, get_test_and_valid_generator
from helpers.df_ops import check_for_leakage, remove_data_leakage
from helpers.io_ops import read_data
from helpers.ai.loss import compute_class_freqs, get_weighted_loss
import pandas as pd 
import seaborn as sns
import tensorflow as tf 
import os
import re
from helpers.text.sort.text_sort import sort_alphanumeric


def fetch_data():
    """
    we will be using the [ChestX-ray8 dataset](https://arxiv.org/abs/1705.02315) 
    which contains 108,948 frontal-view X-ray images of 32,717 unique patients. 
        - Each image in the data set contains multiple text-mined labels identifying 14 different pathological conditions. 
        - These in turn can be used by physicians to diagnose 8 different diseases. 
        - We will use this data to develop a single model that will provide binary classification predictions for each of the 14 labeled pathologies. 
        - In other words it will predict 'positive' or 'negative' for each of the pathologies.
 
    You can download the entire dataset for free [here](https://nihcc.app.box.com/v/ChestXray-NIHCC). 
    - We have used a ~1000 image.

    he dataset includes a CSV file that provides the labels for each X-ray. 

    To make the job a bit easier, we have processed the labels for our small sample and generated three new files to get you started. 
    These three files are:

    1. `nih/train-small.csv`: 875 images from our dataset to be used for training.
    1. `nih/valid-small.csv`: 109 images from our dataset to be used for validation.
    1. `nih/test.csv`: 420 images from our dataset to be used for testing. 
 """   
    train_df = read_data("data/nih/train-small.csv")
    valid_df = read_data("data/nih/valid-small.csv")
    test_df = read_data("data/nih/test-small.csv")

    return train_df, valid_df, test_df

# define function to check for leakage
def check_data_leakage(train_df, valid_df, test_df, patient_col):
    # Check for leakage
    print("leakage between train and valid: {}".format(check_for_leakage(train_df, valid_df, 'PatientId')))
    print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
    print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))

    if check_for_leakage(train_df, valid_df, patient_col):
        print("checking for leakage between train and valid set...")
        train_df, valid_df = remove_data_leakage(train_df, valid_df, patient_col)
    

    if check_for_leakage(train_df, test_df, patient_col):
        print("checking for leakage between train and test set...")
        train_df, test_df = remove_data_leakage(train_df, test_df, patient_col)

    return train_df, valid_df, test_df

def create_model(loss):
    base_model = DenseNet121(weights= WEIGHTS_DIR, include_top=False)
        
    x = base_model.output

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # and a logistic layer
    predictions = Dense(len(LABELS), activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss=  loss)

    print(model.summary())
    return model

def create_callbacks():
    callbacks = [tf.keras.callbacks.ModelCheckpoint(OUTPUT_DIR + 'checkpoints/checkpoint-{epoch}.h5', save_best_only=False, save_weights_only = True,  verbose=1)]
    return callbacks

def create_output_dirs():
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR + 'checkpoints'):
        os.makedirs(OUTPUT_DIR + 'checkpoints')


def load_weights_from_checkpoints(model, checkpoint_path: str) -> int:
        """Given a folder of checkpoints, load the latest checkpoint into the model"""
        if model is None:
            print("ERROR: Model was not initialized or loaded from file")
            return 0

        if os.path.exists(checkpoint_path) == False:
            print("WARNING: Checkpoint directory not found")
            return 0

        checkpoint_files = [
            file for file in os.listdir(checkpoint_path) if file.endswith("." + "h5")
        ]

        max_epoch_number = 0

        if len(checkpoint_files) == 0:
            print(
                "WARNING: Checkpoint directory empty. Path provided: "
                + str(checkpoint_path)
            )
            return max_epoch_number

        checkpoint_files = sort_alphanumeric(checkpoint_files)
        print("------------------------------------------------------")
        print("Available checkpoint files: {}".format(checkpoint_files))
        max_epoch_number = int(re.findall(r"\d+", checkpoint_files[-1][:-3])[0])
        max_epoch_filename = checkpoint_files[-1]

        print("Latest epoch checkpoint file name: {}".format(max_epoch_filename))
        print("Resuming training from epoch: {}".format(int(max_epoch_number) + 1))
        print("------------------------------------------------------")
        model.load_weights(f"{checkpoint_path}/{max_epoch_filename}")
        return model, max_epoch_number

def main():
    # Fetch data
    train_df, valid_df, test_df = fetch_data()
        
    # Check for data leakage
    train_df,valid_df, test_df = check_data_leakage(train_df, valid_df, test_df, "PatientId")

    # Get generators
    train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", LABELS)
    valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", LABELS)

    freq_pos, freq_neg = compute_class_freqs(train_generator.labels)    
    pos_weights = freq_neg
    neg_weights = freq_pos

    create_output_dirs()

    loss =  get_weighted_loss(pos_weights, neg_weights)
    callbacks = create_callbacks()
    model = create_model(loss)
    model, initial_epoch = load_weights_from_checkpoints(model, OUTPUT_DIR + 'checkpoints')
    
    history = model.fit(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 10, 
                              callbacks = callbacks,
                              initial_epoch = initial_epoch)
    

if __name__ == "__main__":
    
    IMAGE_DIR  = "data/nih/images-small/"
    LABELS = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
    
    WEIGHTS_DIR = 'data/nih/models/densenet.hdf5'

    OUTPUT_DIR = "data/output/chest_xray_diagnosis/"

    main()