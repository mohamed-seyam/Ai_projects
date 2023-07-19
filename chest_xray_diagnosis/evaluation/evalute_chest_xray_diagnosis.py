from helpers.evaluations.performance_metrics import (
    get_accuracy, get_prevalence, get_sensitivity, 
    get_specificity, get_ppv, get_npv, get_performance_metrics
)
from helpers.io.io_ops import read_data

import numpy as np




if __name__ == "__main__":
    class_labels = ['Cardiomegaly',
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
    
    # the labels for prediction values in our dataset
    pred_labels = [l + "_pred" for l in class_labels]
    train_results = read_data("data/nih/predictions/train_preds.csv")
    valid_results = read_data("data/nih/predictions/valid_preds.csv")
    
    y = valid_results[class_labels].values
    pred = valid_results[pred_labels].values

    # by these steps we ensure to remove unnecessary columns
    valid_results[np.concatenate([class_labels, pred_labels])].head()

    df_result = get_performance_metrics(y, pred, class_labels)
    print(df_result)
   