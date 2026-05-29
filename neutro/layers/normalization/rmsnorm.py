import numpy as np
from ..base import Layer

class RMSNorm(Layer):
    """
    Root Mean Square Layer Normalization.
    Used in Llama, Qwen, and DeepSeek because we decided that subtracting the mean
    was a luxury we could no longer afford in the 2020s.
    """
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.params['weight'] = np.ones(self.dim)
        super().build(input_shape)

    def forward(self, x, training=False):
        self.x = x
        # RMS = sqrt(mean(x^2) + eps)
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.epsilon)
        self.x_norm = x / self.rms
        return self.x_norm * self.params['weight']

    def backward(self, grad_output):
        # Naive but educational implementation of RMSNorm backward
        # dW
        self.grads['weight'] = np.sum(grad_output * self.x_norm, axis=(0, 1))
        
        # dX
        N = self.dim
        grad_x_norm = grad_output * self.params['weight']
        
        # Backward through: x / sqrt(mean(x^2) + eps)
        # Detailed derivation for the old folks:
        # dx = (grad_x_norm / rms) - (x * sum(grad_x_norm * x) / (N * rms^3))
        
        sum_grad_x = np.sum(grad_x_norm * self.x, axis=-1, keepdims=True)
        dx = (grad_x_norm / self.rms) - (self.x * sum_grad_x / (N * self.rms**3))
        
        return dx
