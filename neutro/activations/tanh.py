import numpy as np
from .base import Activation
from neutro.autograd import ops as autograd_ops

class Tanh(Activation):
    def __call__(self, x):
        return autograd_ops.tanh(x)
    def gradient(self, x):
        return 1 - np.tanh(x) ** 2