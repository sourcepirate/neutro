import numpy as np
from .base import Activation

class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)
    def gradient(self, x):
        return 1 - np.tanh(x)**2
