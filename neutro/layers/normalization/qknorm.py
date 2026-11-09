import numpy as np
from ..base import Layer
from neutro.autograd import Tensor


class QKNorm(Layer):
    """Per-head RMS normalization for Q and K (Qwen3-style).

    Normalizes last dim with learned gain. Input (..., H, D).
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.dim = input_shape[-1]
        # Keep the same parameter name as the other normalization layers.
        self.params['gamma'] = Tensor(np.ones(self.dim))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, x, training=False):
        rms = ((x ** 2).mean(axis=-1, keepdims=True) + self.epsilon).sqrt()
        return (x / rms) * self.params['gamma']
