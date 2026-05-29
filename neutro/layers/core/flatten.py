from ..base import Layer

class Flatten(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape_orig = input_shape
        super().build(input_shape)

    def forward(self, inputs, training=False):
        self.input_shape_orig = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape_orig)
