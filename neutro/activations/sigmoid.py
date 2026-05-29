import numpy as np
from .base import Activation

class Sigmoid(Activation):
    def __call__(self, x):
        self.last_output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.last_output
    def gradient(self, x):
        s = self.__call__(x)
        return s * (1 - s)
