import numpy as np
from ..base import Layer

class UpSampling2D(Layer):
    """
    Upsampling layer for 2D inputs.
    Repeats the rows and columns of the data by size[0] and size[1] respectively.

    Args:
        size: int, or tuple of 2 integers. The upsampling factors for rows and columns.
    """
    def __init__(self, size=(2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        return (batch, h * self.size[0], w * self.size[1], c)

    def forward(self, inputs, training=False):
        self.input_shape_actual = inputs.shape
        # Nearest neighbor upsampling
        return np.repeat(np.repeat(inputs, self.size[0], axis=1), self.size[1], axis=2)

    def backward(self, grad_output):
        # Gradient of upsampling is essentially downsampling by summing
        batch, h, w, c = self.input_shape_actual
        sh, sw = self.size
        
        # Reshape to (batch, h, sh, w, sw, c)
        grad = grad_output.reshape(batch, h, sh, w, sw, c)
        # Sum over the upsampled dimensions
        return grad.sum(axis=(2, 4))
