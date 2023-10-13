"""CallBacks
-------------------------------
- provide some useful functionality during training
- Subclass the tf.keras.callbacks.Callback class
- Useful in understanding a model's internal states and statistics (losses, metrics) during training

Training Specific method
-------------------------------
Class CallBack(object):
    def __init__(self, ):
        self.validation_data = None
        self.model = None 
    
    def on_(train|test|predict)_begin(self, logs=None):
        "called at the begin of fit/evaluate/predict"
    
    def on_(train|test|predict)_end(self, logs=None):
        "called at the end of fit/evaluate/predict"

    def on_(train|test|predict)_batch_begin(self, batch, logs=None):
        "called right before processing a batch during training/testing/predicting"

    def on_(train|test|predict)_batch_end(self, batch, logs=None):
        "called at the end of training/testing/predicting a batch"
      
"""

import tensorflow as tf 

class TrackAccCallback(tf.keras.callbacks.Callback):
        # Define the correct function signature for on_epoch_end
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:                 
                print("\nReached 99% accuracy so cancelling training!")
                
                # Stop training once the above condition is met
                self.model.stop_training = True



