import numpy as np
from .base import Metric

class Accuracy(Metric):
    def __call__(self, y_true, y_pred):
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
        return np.mean(np.round(y_true) == np.round(y_pred))
    def get_name(self):
        return "accuracy"
