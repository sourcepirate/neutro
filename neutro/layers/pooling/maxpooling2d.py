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
    def __init__(self, pool_size=(2, 2), strides=None, data_format='channels_last', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size if isinstance(pool_size, (tuple, list)) else (pool_size, pool_size)
        strides = strides if strides else self.pool_size
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        self.data_format = data_format

    def _shape_to_channels_last(self, shape):
        if self.data_format == 'channels_first':
            batch, c, h, w = shape
            return batch, h, w, c
        return shape

    def _to_channels_last(self, inputs):
        if self.data_format == 'channels_first':
            return inputs.transpose(0, 2, 3, 1)
        return inputs

    def _from_channels_last(self, inputs):
        if self.data_format == 'channels_first':
            return inputs.transpose(0, 3, 1, 2)
        return inputs

    def compute_output_shape(self, input_shape):
        batch, h, w, c = self._shape_to_channels_last(input_shape)
        ph, pw = self.pool_size
        sh, sw = self.strides
        oh = (h - ph) // sh + 1
        ow = (w - pw) // sw + 1
        if self.data_format == 'channels_first':
            return (batch, c, oh, ow)
        return (batch, oh, ow, c)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        inputs_nhwc = self._to_channels_last(inputs)
        batch, h, w, c = inputs_nhwc.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        oh = (h - ph) // sh + 1
        ow = (w - pw) // sw + 1
        
        x = inputs_nhwc.transpose(0, 3, 1, 2).reshape(-1, 1, h, w)
        
        self.x_cols = im2col_indices(x, ph, pw, padding=0, stride=sh)
        self.arg_max = np.argmax(self.x_cols, axis=0)
        out = self.x_cols[self.arg_max, np.arange(self.arg_max.size)]
        
        out = out.reshape(oh, ow, batch, c).transpose(2, 0, 1, 3)
        return self._from_channels_last(out)

    def backward(self, grad_output):
        grad_output_nhwc = self._to_channels_last(grad_output)
        batch, oh, ow, c = grad_output_nhwc.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        _, h, w, _ = self._shape_to_channels_last(self.inputs.shape)
        
        dout = grad_output_nhwc.transpose(1, 2, 0, 3).flatten()
        
        dx_cols = np.zeros_like(self.x_cols)
        dx_cols[self.arg_max, np.arange(self.arg_max.size)] = dout
        
        dx = col2im_indices(dx_cols, (batch * c, 1, h, w), ph, pw, padding=0, stride=sh)
        return self._from_channels_last(dx.reshape(batch, c, h, w).transpose(0, 2, 3, 1))
