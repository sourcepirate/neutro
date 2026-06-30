from ..base import Layer

class Flatten(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape_orig = input_shape
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        import numpy as np
        return (input_shape[0], int(np.prod(input_shape[1:])))

    def forward(self, inputs, training=False):
        return inputs.reshape(inputs.shape[0], -1)