import numpy as np
from ..base import Layer
from ...utils.conv_utils import im2col_indices, col2im_indices

class MaxPooling2D(Layer):
    """
    Max pooling operation for 2D spatial data.

    Args:
        pool_size: Integer or tuple of 2 integers, window size over which to take the maximum.
        strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
    """
    def __init__(self, pool_size=(2, 2), strides=None, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size if isinstance(pool_size, (tuple, list)) else (pool_size, pool_size)
        self.strides = strides if strides else self.pool_size

    def forward(self, inputs, training=False):
        self.inputs = inputs
        batch, h, w, c = inputs.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        oh = (h - ph) // sh + 1
        ow = (w - pw) // sw + 1
        
        # Internally use (N*C, 1, H, W) to use im2col as it supports multichannel but we want pool per channel
        x = inputs.transpose(0, 3, 1, 2).reshape(-1, 1, h, w)
        
        self.x_cols = im2col_indices(x, ph, pw, padding=0, stride=sh)
        self.arg_max = np.argmax(self.x_cols, axis=0)
        out = self.x_cols[self.arg_max, np.arange(self.arg_max.size)]
        
        out = out.reshape(oh, ow, batch, c).transpose(2, 0, 1, 3)
        return out

    def backward(self, grad_output):
        batch, oh, ow, c = grad_output.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        h, w = self.inputs.shape[1], self.inputs.shape[2]
        
        dout = grad_output.transpose(1, 2, 0, 3).flatten()
        
        dx_cols = np.zeros_like(self.x_cols)
        dx_cols[self.arg_max, np.arange(self.arg_max.size)] = dout
        
        dx = col2im_indices(dx_cols, (batch * c, 1, h, w), ph, pw, padding=0, stride=sh)
        return dx.reshape(batch, c, h, w).transpose(0, 2, 3, 1)
