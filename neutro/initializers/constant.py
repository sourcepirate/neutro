import numpy as np
from .base import Initializer

class Zeros(Initializer):
    def __call__(self, shape):
        return np.zeros(shape)

class Ones(Initializer):
    def __call__(self, shape):
        return np.ones(shape)
