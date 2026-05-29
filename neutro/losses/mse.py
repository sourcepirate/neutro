import numpy as np
from .base import Loss

class MeanSquaredError(Loss):
    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    def gradient(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size
