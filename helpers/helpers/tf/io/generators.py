"""Generators for training and validation data sets
- This file contains the code for creating the generators for the training and validation data sets
- The generators are used to load the data in batches
- The generators also perform data augmentation
- Flow from directory is used to create the generators
- Or Flow from dataframe is used to create the generators
"""

import tensorflow as tf

def train_val_generators(training_dir:str, 
                         validation_dir:str,
                         target_size : tuple = (150,150),
                         train_batch_size :int  = 50,
                         validation_batch_size:int = 30,

                         )->tuple[tf.keras.preprocessing.image.ImageDataGenerator, 
                                  tf.keras.preprocessing.image.ImageDataGenerator]:
    """Creates the training and validation data generators"""

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,
                                        rotation_range=40,
                                        width_shift_range=.2,
                                        height_shift_range=.2,
                                        shear_range=.2,
                                        zoom_range=.2,
                                        horizontal_flip=True,
                                        fill_mode="nearest")

    train_generator = train_datagen.flow_from_directory(directory=training_dir,
                                                        batch_size=train_batch_size,
                                                        class_mode="binary",
                                                        target_size= target_size)

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1/255.
    )

    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                    batch_size=validation_batch_size,
                                                                    class_mode="binary",
                                                                    target_size=target_size)
    return train_generator, validation_generator


import tensorflow as tf

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
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
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
    raw_train_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=image_dir, 
        x_col="Image", 
        y_col=y_cols, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
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