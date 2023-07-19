import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

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
    """This function plot the distribution of pixel values for the original image."""
    sns.distplot(raw_img)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("# Pixels in Image")
    plt.title("Distribution of Pixel intensity in the image!")
    plt.show()

def compare_pixel_distribution(raw_img: np.array, raw_img_n: np.array)-> plt.figure:
    """This function plot the distribution of pixel values for the original image and the normalized image."""
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
    plt.title("Comparison between pixel distribution of the original image and the normalized image!")
    plt.xlabel("Pixel value")
    plt.ylabel("# of Pixels")
    plt.show()

def plot_class_counts(df: pd.DataFrame) -> plt.figure:
    """This function plot the distribution of classes in the dataset the input is a data frame with the following format:
    |   Class     | Count |
    |-------------|-------|
    |   Class A   |   10  |
    |   Class B   |   20  |
    |   Class C   |   30  |
    |   Class D   |   40  |
    |   Class E   |   50  |
    """    
    sns.barplot(x = df.values, y = df.index, color='b')
    plt.title('Distribution of Classes for Training Dataset', fontsize=15)
    plt.xlabel('Number of Patients', fontsize=15)
    plt.ylabel('Diseases', fontsize=15)
    plt.show()

