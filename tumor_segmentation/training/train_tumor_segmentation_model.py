import json 
import os 

import tensorflow as tf 
from helpers.ai.io.checkpoints import load_weights_from_checkpoints


def create_callbacks():
    callbacks = [tf.keras.callbacks.ModelCheckpoint(OUTPUT_DIR + 'checkpoints/checkpoint-{epoch}.h5', save_best_only=False, save_weights_only = True,  verbose=1)]
    return callbacks

def create_output_dirs():
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR + 'checkpoints'):
        os.makedirs(OUTPUT_DIR + 'checkpoints')


    
def fetch_data():
    from helpers.ai.io.volume_io import VolumeDataGenerator
    # Get generators for training and validation sets
    train_generator = VolumeDataGenerator(config["train"], base_dir + "train/", batch_size=3, dim=(160, 160, 16), verbose=0)
    valid_generator = VolumeDataGenerator(config["valid"], base_dir + "valid/", batch_size=3, dim=(160, 160, 16), verbose=0)

    return train_generator, valid_generator

def create_model():
    from helpers.ai.models import unet_model_3d
    from helpers.ai.loss import soft_dice_loss
    from helpers.ai.metrics import dice_coefficient
    
    model = unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])
    print(model.summary())
    return model 

def main():
    # fetch data 
    train_generator, valid_generator = fetch_data()

    # create model 
    model = create_model()

    # create callbacks 
    callbacks = create_callbacks()
    # start training
    model.fit(train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            use_multiprocessing=True,
            validation_data=valid_generator,
            validation_steps=validation_steps,
            callbacks = callbacks)


if __name__ == "__main__":
    HOME_DIR = "data/BraTS-Data/"
    DATA_DIR = HOME_DIR
    base_dir = HOME_DIR + "processed/"
    OUTPUT_DIR = "data/output/tumor_segmentation/"

    # config file determine the cases that we will work on to limit the
    # the usage 
    with open(base_dir + "config.json") as json_file:
        config = json.load(json_file)

    steps_per_epoch = 20
    n_epochs=10
    validation_steps = 20
    
    main()