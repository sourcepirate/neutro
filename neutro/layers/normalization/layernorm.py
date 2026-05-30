import numpy as np
from ..base import Layer

class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.params['gamma'] = np.ones(input_shape[-1])
        self.params['beta'] = np.zeros(input_shape[-1])
        super().build(input_shape)

    def forward(self, x, training=False):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return self.params['gamma'] * self.x_norm + self.params['beta']

    def backward(self, grad_output):
        N = grad_output.shape[-1]
        self.grads['gamma'] = np.sum(grad_output * self.x_norm, axis=tuple(range(len(grad_output.shape)-1)))
        self.grads['beta'] = np.sum(grad_output, axis=tuple(range(len(grad_output.shape)-1)))
        dx_norm = grad_output * self.params['gamma']
        std_inv = 1.0 / np.sqrt(self.var + self.epsilon)
        dx = (1.0 / N) * std_inv * (N * dx_norm - np.sum(dx_norm, axis=-1, keepdims=True) - self.x_norm * np.sum(dx_norm * self.x_norm, axis=-1, keepdims=True))
        return dx
