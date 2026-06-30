import numpy as np
from ..base import Layer
from neutro.autograd import Tensor


class GroupNormalization(Layer):
    def __init__(self, groups=32, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        if dim % self.groups != 0:
            raise ValueError(f"Number of channels ({dim}) must be divisible by groups ({self.groups})")
        self.params['gamma'] = Tensor(np.ones((1, 1, 1, dim)))
        self.params['beta'] = Tensor(np.zeros((1, 1, 1, dim)))
        super().build(input_shape)

    def forward(self, x, training=False):
        batch, h, w, c = x.shape
        g = self.groups
        x_reshaped = x.reshape(batch, h, w, g, c // g)
        mean = x_reshaped.mean(axis=(1, 2, 4), keepdims=True)
        var = ((x_reshaped - mean) ** 2).mean(axis=(1, 2, 4), keepdims=True)
        x_norm = (x_reshaped - mean) / (var + self.epsilon).sqrt()
        x_norm = x_norm.reshape(batch, h, w, c)
        return self.params['gamma'] * x_norm + self.params['beta']