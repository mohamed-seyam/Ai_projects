
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score, roc_curve


def true_positives(y:np.ndarray, pred:np.ndarray, th:float =.5):
    """
    Returns the true positives for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    pred_threshold =  pred >= th 
    tp = np.sum((pred_threshold == 1) & (y == 1))
    return tp 

def true_negatives(y:np.ndarray, pred:np.ndarray, th:float =.5):
    """
    Returns the true negatives for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    pred_threshold =  pred >= th 
    tn = np.sum((pred_threshold == 0) & (y == 0))
    return tn

def false_positives(y:np.ndarray, pred:np.ndarray, th:float =.5):
    """
    Returns the false positives for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    pred_threshold =  pred >= th 
    fp = np.sum((pred_threshold == 1) & (y == 0))
    return fp

def false_negatives(y:np.ndarray, pred:np.ndarray, th:float =.5):
    """
    Returns the false negatives for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    pred_threshold =  pred >= th 
    fn = np.sum((pred_threshold == 0) & (y == 1))
    return fn

def get_accuracy(y:np.ndarray, pred:np.ndarray, th:float =.5):
    """
    Returns the accuracy for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    tp = true_positives(y, pred, th)
    tn = true_negatives(y, pred, th)
    fp = false_positives(y, pred, th)
    fn = false_negatives(y, pred, th)

    return (tp + tn) / (tp + tn + fp + fn)

def get_prevalence(y:np.ndarray):
    """
    Returns the prevalence of the positive class
    """
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"

    return np.mean(y == 1)

def get_sensitivity(y:np.ndarray, pred:np.ndarray, th:float =.5):
    """
    Returns the sensitivity for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    tp = true_positives(y, pred, th)
    fn = false_negatives(y, pred, th)

    return tp / (tp + fn)

def get_specificity(y:np.ndarray, pred:np.ndarray, th:float =.5):
    """
    Returns the specificity for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    tn = true_negatives(y, pred, th)
    fp = false_positives(y, pred, th)

    return tn / (tn + fp)

def get_tpr(y:np.ndarray, pred:np.ndarray, th:float = .5):
    """
    Returns the true positive rate for a given threshold
    """
    return get_sensitivity(y, pred, th)

def get_fpr(y:np.ndarray, pred:np.ndarray, th:float = .5):
    """
    Returns the false positive rate for a given threshold
    """
    return 1 - get_specificity(y, pred, th)

def get_precision(y:np.ndarray, pred:np.ndarray, th:float = .5):
    """
    Returns the precision for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    tp = true_positives(y, pred, th)
    fp = false_positives(y, pred, th)

    return tp / (tp + fp)

def get_ppv(y:np.ndarray, pred:np.ndarray, th:float = .5):
    """
    Returns the positive predictive value for a given threshold
    """
    return get_precision(y, pred, th)

def get_npv(y:np.ndarray, pred:np.ndarray, th:float = .5):
    """
    Return the negative predictive value for a given threshold
    """
    assert len(y) == len(pred), "y and pred must have the same length"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert isinstance(pred, np.ndarray), "pred must be a numpy ndarray"

    tn = true_negatives(y, pred, th)
    fn = false_negatives(y, pred, th)

    return tn / (tn + fn)

def get_performance_metrics(y, pred, class_labels, tp= true_positives,
                            tn=true_negatives, fp=false_positives,
                            fn=false_negatives,
                            acc=None, prevalence=None, spec=None,
                            sens=None, ppv=None, npv=None, auc=None, f1=None,
                            thresholds=[]):
    # # this line is to suppress the warning message that appears during using loc function
    # pd.set_option("mode.chained_assignment", None)

    if len(thresholds) != len(class_labels):
        thresholds = [.5] * len(class_labels)

    columns = ["", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
               "Sensitivity",
               "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = class_labels[i]
        df.loc[i][1] = round(tp(y[:, i], pred[:, i]),
                             3) if tp != None else "Not Defined"
        df.loc[i][2] = round(tn(y[:, i], pred[:, i]),
                             3) if tn != None else "Not Defined"
        df.loc[i][3] = round(fp(y[:, i], pred[:, i]),
                             3) if fp != None else "Not Defined"
        df.loc[i][4] = round(fn(y[:, i], pred[:, i]),
                             3) if fn != None else "Not Defined"
        df.loc[i][5] = round(acc(y[:, i], pred[:, i], thresholds[i]),
                             3) if acc != None else "Not Defined"
        df.loc[i][6] = round(prevalence(y[:, i]),
                             3) if prevalence != None else "Not Defined"
        df.loc[i][7] = round(sens(y[:, i], pred[:, i], thresholds[i]),
                             3) if sens != None else "Not Defined"
        df.loc[i][8] = round(spec(y[:, i], pred[:, i], thresholds[i]),
                             3) if spec != None else "Not Defined"
        df.loc[i][9] = round(ppv(y[:, i], pred[:, i], thresholds[i]),
                             3) if ppv != None else "Not Defined"
        df.loc[i][10] = round(npv(y[:, i], pred[:, i], thresholds[i]),
                              3) if npv != None else "Not Defined"
        df.loc[i][11] = round(auc(y[:, i], pred[:, i]),
                              3) if auc != None else "Not Defined"
        df.loc[i][12] = round(f1(y[:, i], pred[:, i] > thresholds[i]),
                              3) if f1 != None else "Not Defined"
        df.loc[i][13] = round(thresholds[i], 3)

    df = df.set_index("")
    return df



def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals

