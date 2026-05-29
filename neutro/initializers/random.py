import numpy as np
from .base import Initializer

class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev
    def __call__(self, shape):
        return np.random.normal(self.mean, self.stddev, shape)
