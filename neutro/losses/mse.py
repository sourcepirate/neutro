import numpy as np
from .base import Loss
from neutro.autograd import Tensor, as_tensor


class MeanSquaredError(Loss):
    def __call__(self, y_true, y_pred):
        y_true = as_tensor(y_true)
        return ((y_pred - y_true) ** 2).mean()

    def gradient(self, y_true, y_pred):
        y_true = as_tensor(y_true)
        return 2.0 * (y_pred - y_true) / y_true.data.size