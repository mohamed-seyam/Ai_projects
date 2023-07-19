import os 
import numpy as np 
import pandas as pd
import keras.utils as image

def get_mean_std_per_batch(image_dir: str , df : pd.DataFrame, H: int =320, W : int =320):
    """Get mean and std per batch of images.
    Args:
        image_dir: directory with images
        df: dataframe with data
        H: image height
        W: image width
    Returns:
        tuple with mean and std
    """
    sample_data = []
    for img in df.sample(100)["Image"].values:
        image_path = os.path.join(image_dir, img)
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data, axis=(0, 1, 2, 3))
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
    return mean, std
