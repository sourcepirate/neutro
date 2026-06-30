import numpy as np
from ..base import Layer
from neutro.autograd import Tensor


class RMSNorm(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.params['weight'] = Tensor(np.ones(self.dim))
        super().build(input_shape)

    def forward(self, x, training=False):
        rms = ((x ** 2).mean(axis=-1, keepdims=True) + self.epsilon).sqrt()
        x_norm = x / rms
        return x_norm * self.params['weight']