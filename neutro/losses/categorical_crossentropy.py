import numpy as np
from .base import Loss

class CategoricalCrossentropy(Loss):
    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
    def gradient(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Normalize by total number of samples across all dimensions except last
        n_samples = np.prod(y_true.shape[:-1])
        return - (y_true / y_pred) / n_samples
