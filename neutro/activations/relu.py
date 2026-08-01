import numpy as np
from .base import Activation
from neutro.autograd import ops as autograd_ops


class ReLU(Activation):
    """Rectified Linear Unit activation."""

    def __call__(self, x):
        return autograd_ops.relu(x)

    def gradient(self, x):
        return (x > 0).astype(float)
