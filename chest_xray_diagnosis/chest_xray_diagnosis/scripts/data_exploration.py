import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

from helpers.test import (
    read_data,
    investigate_single_image,
    investigate_pixel_value_distribution,
    compare_pixel_distribution,
    check_for_leakage,
    visualize_images,
    explore_data
)

def visualize_images(df):
    
    # Extract numpy values from Image column in data frame
    images = df["Image"].values
    # Extract 9 random images from it
    random_images = [np.random.choice(images) for i in range(9)]

    # Adjust the size of your images
    plt.figure(figsize=(20,10))
    # Iterate and plot random images
    for i in range(9):
        plt.subplot(3,3, i+1)
        img = plt.imread(os.path.join(image_dir + random_images[i]))
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()
    plt.show()   


def explore_data(df: pd.DataFrame):
   
    # 1.1 Data Types and Null Values Check
    print(df.head())
    print(df.info())

    # 1.2 Unique IDs Check
    print(f'Total Patient ids are : {df["PatientId"].count()}')
    print(f'Unique Patient ids are : {df["PatientId"].value_counts().shape[0]}')
    
    # 1.3 Data Labels 
    columns = list(df.keys())
    print(f"Columns of Data frame: \n {columns}")

    # Remove unnecessary elements
    columns.remove("PatientId")
    columns.remove("Image")
    print(f"There are {len(columns)} columns represent labels for conditions! {columns}")


    # Print number of positive labels for every condition
    for column in columns:
        print(f"the class {column} has {df[column].sum()} Samples")


if __name__ == "__main__":
    # Location of img dir 
    image_dir = "data/nih/images-small/"
    # 1. Exploration
    train_df = read_data("./data/nih/train-small.csv")
    
    print(f"There are {train_df.shape[0]} Rows and {train_df.shape[1]} Columns in the data Frame!")
    
    # explore_data(train_df)
    
    # 1.4 Data Visualization

    # print('Display Random Images')
    # visualize_images(df=train_df)

    #1.5 Investigating a Single Image
    img = train_df.Image[0]
    raw_img = plt.imread(os.path.join(image_dir, img))
    # investigate_single_image(raw_img, title="Chest Xray Image")
    
    # 1.6 Investigating Pixel Value Distribution
    img = train_df.Image[0]
    raw_img = plt.imread(os.path.join(image_dir, img))
    investigate_pixel_value_distribution(raw_img)
    # 2. Image Preprocessing in Keras
    
    # Import data generator from keras 
    from keras.preprocessing.image import ImageDataGenerator
    image_generator = ImageDataGenerator( 
        samplewise_center= True, # each img have 0 mean
        samplewise_std_normalization= True # each img have 1 std deviation 
    )

    # 2.1 Standardization

    # flow from specified directory with specified batch size and target size 
    generator = image_generator.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="Image", # Features
        # let's say we build model for mass detection 
        y_col="Mass", # labels
        class_mode="raw", # mass label should be in dataframe 
        batch_size= 1, # images per batch 
        target_size= (320,320),
        shuffle=False
    )

    # Plot Processed Image 
    generated_imgs, label = generator.__getitem__(0)
    raw_img_n = generated_imgs[0]
    # investigate_single_image(raw_img_n, title="Chest Xray Image")
    
    # Include a histogram of the distribution of the pixels
    compare_pixel_distribution(raw_img, raw_img_n)
