import numpy as np
from .base import Metric

class SparseAccuracy(Metric):
    def __call__(self, y_true, y_pred):
        # y_true: (N,) or (N, S) with integer labels
        # y_pred: (N, C) or (N, S, C) with probabilities
        y_pred_labels = np.argmax(y_pred, axis=-1)
        return np.mean(y_true == y_pred_labels)
        
    def get_name(self):
        return "sparse_accuracy"
