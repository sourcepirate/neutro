import numpy as np
from .base import Activation
from neutro.autograd import ops as autograd_ops

class Sigmoid(Activation):
    def __call__(self, x):
        return autograd_ops.sigmoid(x)
    def gradient(self, x):
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return s * (1 - s)