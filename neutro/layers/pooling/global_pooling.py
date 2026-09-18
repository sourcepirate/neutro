from ..base import Layer
from neutro.autograd import Tensor, ops as autograd_ops


class GlobalAveragePooling2D(Layer):
    def __init__(self, data_format='channels_last', **kwargs):
        super().__init__(**kwargs)
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        channels = input_shape[1] if self.data_format == 'channels_first' else input_shape[-1]
        return (input_shape[0], channels)

    def forward(self, inputs, training=False):
        if self.data_format == 'channels_first':
            return inputs.mean(axis=(2, 3))
        return inputs.mean(axis=(1, 2))


class GlobalMaxPooling2D(Layer):
    def __init__(self, data_format='channels_last', **kwargs):
        super().__init__(**kwargs)
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        channels = input_shape[1] if self.data_format == 'channels_first' else input_shape[-1]
        return (input_shape[0], channels)

    def forward(self, inputs, training=False):
        if self.data_format == 'channels_first':
            x = inputs.transpose(0, 2, 3, 1)
        else:
            x = inputs
        batch, h, w, c = x.shape
        x_flat = x.reshape(batch, h * w, c)
        return autograd_ops.max_op(x_flat, axis=1)