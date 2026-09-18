import numpy as np
from ..base import Layer
from neutro.autograd import Tensor


class BatchNormalization(Layer):
    def __init__(self, momentum=0.99, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None

    def build(self, input_shape):
        dim = input_shape[-1]
        self.params['gamma'] = Tensor(np.ones(dim))
        self.params['beta'] = Tensor(np.zeros(dim))
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        super().build(input_shape)

    def forward(self, x, training=False):
        if training:
            mean = x.mean(axis=tuple(range(x.ndim - 1)))
            var = ((x - mean) ** 2).mean(axis=tuple(range(x.ndim - 1)))
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.data
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.data
            x_norm = (x - mean) / (var + self.epsilon).sqrt()
        else:
            x_norm = (x - Tensor(self.running_mean)) / Tensor(np.sqrt(self.running_var + self.epsilon))
        return self.params['gamma'] * x_norm + self.params['beta']