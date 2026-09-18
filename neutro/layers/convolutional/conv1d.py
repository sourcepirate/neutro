from ..base import Layer
from ...initializers import get as get_initializer
from ...activations import get as get_activation
from neutro.autograd import Tensor
from neutro.autograd.custom_ops import Conv1DFunction


class Conv1D(Layer):
    """1-D convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,)
        )
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides,)
        self.padding = padding
        self.activation = get_activation(activation)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

    def build(self, input_shape):
        _, steps, c = input_shape
        k = self.kernel_size[0]
        self.params['W'] = Tensor(self.kernel_initializer((k, c, self.filters)))
        self.params['b'] = Tensor(self.bias_initializer((self.filters,)))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        batch, steps, c = input_shape
        k = self.kernel_size[0]
        s = self.strides[0]
        padding = 0
        if self.padding == "same":
            padding = (k - 1) // 2
        out_steps = (steps + 2 * padding - k) // s + 1
        return (batch, out_steps, self.filters)

    def forward(self, inputs, training=False):
        out = Conv1DFunction.apply(
            inputs, self.params["W"], self.params["b"],
            self.kernel_size, self.strides, self.padding,
        )
        if self.activation:
            out = self.activation(out)
        return out