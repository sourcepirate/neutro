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
        # dW: sum over all axes except the last (feature) axis so the result
        # has the same shape as params['weight'] regardless of input rank
        # (works for 2-D (batch, dim), 3-D (batch, seq, dim), etc.)
        feature_axes = tuple(range(len(grad_output.shape) - 1))
        self.grads['weight'] = np.sum(grad_output * self.x_norm, axis=feature_axes)

        # dX
        N = self.dim
        grad_x_norm = grad_output * self.params['weight']

        # Backward through: x / sqrt(mean(x^2) + eps)
        # Derivation:
        #   rms = sqrt(mean(x^2) + eps),  d_rms/dx_j = x_j / (N * rms)
        #   d_xnorm_i/dx_j = delta_ij/rms - x_i*x_j / (N * rms^3)
        #   dL/dx_j = grad_x_norm_j/rms - x_j * sum(grad_x_norm*x) / (N * rms^3)

        sum_grad_x = np.sum(grad_x_norm * self.x, axis=-1, keepdims=True)
        dx = (grad_x_norm / self.rms) - (self.x * sum_grad_x / (N * self.rms**3))

        return dx
