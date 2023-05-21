import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

def read_data(file_path:str)-> pd.DataFrame:
    return pd.read_csv(file_path)

def investigate_single_image(raw_img, title = ""):
    """This function display image with information about it"""
    plt.imshow(raw_img, cmap="gray")
    plt.colorbar()
    plt.title(title)
    print(f"The dimensions of the image are {raw_img.shape[0]} pixels width and {raw_img.shape[1]} pixels height, one single color channel")
    print(f"The maximum pixel value is {raw_img.max():.4f} and the minimum is {raw_img.min():.4f}")
    print(f"The mean value of the pixels is {raw_img.mean():.4f} and the standard deviation is {raw_img.std():.4f}")
    plt.show()


def investigate_pixel_value_distribution(raw_img):
    sns.histplot(data = raw_img)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("# Pixels in Image")
    plt.title("Distribution of Pixel intensity in the image!")
    plt.show()

import seaborn as sns

def compare_pixel_distribution(raw_img, raw_img_n):
    sns.set()
    plt.figure(figsize=(10, 7))
    sns.histplot(raw_img.ravel(),
                 label=f'Original Image: mean {np.mean(raw_img):.4f} - Standard Deviation {np.std(raw_img):.4f} \n '
                       f'Min pixel value {np.min(raw_img):.4} - Max pixel value {np.max(raw_img):.4}',
                 color='blue',)
    sns.histplot(raw_img_n.ravel(),
                 label=f'Generated Image: mean {np.mean(raw_img_n):.4f} - Standard Deviation {np.std(raw_img_n):.4f} \n'
                       f'Min pixel value {np.min(raw_img_n):.4} - Max pixel value {np.max(raw_img_n):.4}',
                 color='red', )

    plt.legend()
    plt.title("Comparsion between pixel distribution of the original image and the normalized image!")
    plt.xlabel("Pixel value")
    plt.ylabel("# of Pixels")
    plt.show()

def check_for_leakage(df1 : pd.DataFrame , df2: pd.DataFrame, patient_col:str)-> bool:
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
    
    df1_patients_unique =set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)
    
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    if len(patients_in_both_groups) > 0 :
        leakage = True 
    else : 
        leakage = False
    # boolean (true if there is at least 1 patient in both groups)
   
    return leakage

def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator

def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator