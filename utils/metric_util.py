from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
)
from .log_util import logger
import numpy as np


def calc_metrics(y_true, y_score, threshold = 0.5):
    if not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)
    y_score = y_score > threshold
    accuracy = accuracy_score(y_true, y_score)
    f1 = f1_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_score)
    precision = precision_score(y_true, y_score)
    recall = recall_score(y_true, y_score)
    return accuracy, f1, mcc, precision, recall


def calc_f1_precision_recall(y_true, y_predict):
    """  """
    # accuracy = accuracy_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    return f1, precision, recall


def find_threshold(y_true, y_score, alpha = 0.05):
    """ return threshold when fpr <= 0.05 """
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    for i, _fpr in enumerate(fpr):
        if _fpr > alpha:
            return thresh[i-1]


def roc(y_true, y_score):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    return roc, fpr, tpr


def calc_metrics_at_thresholds(y_true, y_pred_probability, thresholds=None, default_threshold=None):
    """  """
    if default_threshold:
        accuracy, f1, mcc, precision, recall = calc_metrics(y_true, y_pred_probability, default_threshold)
        logger.info(f'default_threshold {default_threshold}')
        logger.info(f"accuracy: {accuracy}\nf1: {f1}\nmcc: {mcc}\nprecision: {precision}\nrecall: {recall}")

    if not thresholds: return
    for threshold in thresholds:
        accuracy, f1, mcc, precision, recall = calc_metrics(y_true, y_pred_probability, threshold)
        logger.info(f'threshold {threshold}')
        logger.info(f"accuracy: {accuracy}\nf1: {f1}\nmcc: {mcc}\nprecision: {precision}\nrecall: {recall}")

        if default_threshold:
            # Extra test on the threshold value
            mid_threshold =  (threshold + default_threshold) / 2
            accuracy, f1, mcc, precision, recall = calc_metrics(y_true, y_pred_probability, mid_threshold)
            logger.info(f'mid_threshold {mid_threshold} which is the mean of threshold and default threshold')
            logger.info(f"accuracy: {accuracy}\nf1: {f1}\nmcc: {mcc}\nprecision: {precision}\nrecall: {recall}")
            