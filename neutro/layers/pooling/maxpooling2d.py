import numpy as np
from ..base import Layer
from neutro.autograd import Tensor
from neutro.autograd.custom_ops import MaxPool2DFunction


class MaxPooling2D(Layer):
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
        inputs_nhwc = self._to_channels_last(inputs)
        out = MaxPool2DFunction.apply(inputs_nhwc, self.pool_size, self.strides)
        return self._from_channels_last(out)