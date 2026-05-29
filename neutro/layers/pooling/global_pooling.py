import numpy as np
from ..base import Layer

class GlobalAveragePooling2D(Layer):
    """
    Global average pooling operation for spatial data.
    """
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def forward(self, inputs, training=False):
        self.input_shape_internal = inputs.shape
        # inputs: (batch, h, w, c)
        return np.mean(inputs, axis=(1, 2))

    def backward(self, grad_output):
        # grad_output: (batch, c)
        batch, h, w, c = self.input_shape_internal
        return (grad_output[:, None, None, :] * np.ones((batch, h, w, c))) / (h * w)

class GlobalMaxPooling2D(Layer):
    """
    Global max pooling operation for spatial data.
    """
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def forward(self, inputs, training=False):
        self.inputs = inputs
        # inputs: (batch, h, w, c)
        self.max_indices = np.argmax(inputs.reshape(inputs.shape[0], -1, inputs.shape[-1]), axis=1)
        return np.max(inputs, axis=(1, 2))

    def backward(self, grad_output):
        batch, h, w, c = self.inputs.shape
        dx = np.zeros_like(self.inputs)
        # This is a bit slow with loops, but correct. For global max pool it's one max per channel per batch.
        for b in range(batch):
            for channel in range(c):
                idx = self.max_indices[b, channel]
                ih, iw = divmod(idx, w)
                dx[b, ih, iw, channel] = grad_output[b, channel]
        return dx
