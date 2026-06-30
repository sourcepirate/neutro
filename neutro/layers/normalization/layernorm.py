import numpy as np
from ..base import Layer
from neutro.autograd import Tensor


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.params['gamma'] = Tensor(np.ones(input_shape[-1]))
        self.params['beta'] = Tensor(np.zeros(input_shape[-1]))
        super().build(input_shape)

    def forward(self, x, training=False):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / (var + self.epsilon).sqrt()
        return self.params['gamma'] * x_norm + self.params['beta']