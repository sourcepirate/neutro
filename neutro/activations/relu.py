import numpy as np
from .base import Activation

class ReLU(Activation):
    def __call__(self, x):
        self.last_x = x
        return np.maximum(0, x)
    def gradient(self, x):
        return (x > 0).astype(float)
