import numpy as np
from .base import Metric

class Recall(Metric):
    def __call__(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else np.round(y_true)
        y_pred_labels = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else np.round(y_pred)
        tp = np.sum((y_true_labels == 1) & (y_pred_labels == 1))
        fn = np.sum((y_true_labels == 1) & (y_pred_labels == 0))
        return tp / (tp + fn + 1e-15)
    def get_name(self):
        return "recall"
