from ..base import Layer
from neutro.autograd import Tensor, ops as autograd_ops


class UpSampling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        return (batch, h * self.size[0], w * self.size[1], c)

    def forward(self, inputs, training=False):
        x = autograd_ops.repeat_op(inputs, self.size[0], axis=1)
        return autograd_ops.repeat_op(x, self.size[1], axis=2)