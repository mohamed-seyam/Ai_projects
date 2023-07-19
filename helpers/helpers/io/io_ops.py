import os 

import numpy as np
import pandas as pd
import keras.utils as image

from helpers.preprocessing.preprocessing import get_mean_std_per_batch



def read_data(file_path:str)-> pd.DataFrame:
    return pd.read_csv(file_path)


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean, std = get_mean_std_per_batch(image_dir, df, H=H, W=W)
    img_path = os.path.join(image_dir, img)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x
