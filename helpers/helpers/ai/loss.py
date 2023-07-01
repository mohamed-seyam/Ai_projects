import numpy as np
import pandas as pd
import keras.backend as K

def compute_class_freqs(labels):
    """
    Compute positive and negative frequencies for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes) (hot encoded)
    Returns:
        positive_frequencies (np.array): array of positive frequencies for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequencies for each
                                         class, size (num_classes)
    """
    # total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels , axis = 0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies



def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        y_true = K.cast(y_true, dtype=K.floatx())  # Convert y_true to float32
        y_pred = K.cast(y_pred, dtype=K.floatx())  # Convert y_pred to float32
        loss = 0.0
        
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += K.mean(-(pos_weights[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon) + neg_weights[i] * (1-y_true[:,i]) * K.log(1-y_pred[:,i] + epsilon)))
        return loss

    return weighted_loss