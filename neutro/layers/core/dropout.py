import numpy as np
from ..base import Layer

class Dropout(Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.mask = None

    def forward(self, inputs, training=False):
        if not training or self.rate == 0:
            return inputs
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
        return inputs * self.mask

    def backward(self, grad_output):
        if self.mask is None:
            return grad_output
        return grad_output * self.mask
