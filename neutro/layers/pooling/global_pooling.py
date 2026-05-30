import numpy as np
from ..base import Layer

class GlobalAveragePooling2D(Layer):
    """
    Global average pooling operation for spatial data.
    """
    def __init__(self, data_format='channels_last', **kwargs):
        super().__init__(**kwargs)
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        channels = input_shape[1] if self.data_format == 'channels_first' else input_shape[-1]
        return (input_shape[0], channels)

    def forward(self, inputs, training=False):
        self.input_shape_internal = inputs.shape
        if self.data_format == 'channels_first':
            return np.mean(inputs, axis=(2, 3))
        return np.mean(inputs, axis=(1, 2))

    def backward(self, grad_output):
        if self.data_format == 'channels_first':
            batch, c, h, w = self.input_shape_internal
            return (grad_output[:, :, None, None] * np.ones((batch, c, h, w))) / (h * w)
        batch, h, w, c = self.input_shape_internal
        return (grad_output[:, None, None, :] * np.ones((batch, h, w, c))) / (h * w)

class GlobalMaxPooling2D(Layer):
    """
    Global max pooling operation for spatial data.
    """
    def __init__(self, data_format='channels_last', **kwargs):
        super().__init__(**kwargs)
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        channels = input_shape[1] if self.data_format == 'channels_first' else input_shape[-1]
        return (input_shape[0], channels)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        if self.data_format == 'channels_first':
            inputs_nhwc = inputs.transpose(0, 2, 3, 1)
        else:
            inputs_nhwc = inputs
        self.max_indices = np.argmax(inputs_nhwc.reshape(inputs_nhwc.shape[0], -1, inputs_nhwc.shape[-1]), axis=1)
        return np.max(inputs_nhwc, axis=(1, 2))

    def backward(self, grad_output):
        if self.data_format == 'channels_first':
            batch, c, h, w = self.inputs.shape
            dx_nhwc = np.zeros((batch, h, w, c), dtype=self.inputs.dtype)
        else:
            batch, h, w, c = self.inputs.shape
            dx_nhwc = np.zeros_like(self.inputs)
        for b in range(batch):
            for channel in range(c):
                idx = self.max_indices[b, channel]
                ih, iw = divmod(idx, w)
                dx_nhwc[b, ih, iw, channel] = grad_output[b, channel]
        if self.data_format == 'channels_first':
            return dx_nhwc.transpose(0, 3, 1, 2)
        return dx_nhwc
