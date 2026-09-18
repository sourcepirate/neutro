from ..base import Layer
from ...initializers import get as get_initializer
from ...activations import get as get_activation
from neutro.autograd import Tensor
from neutro.autograd.custom_ops import Conv2DFunction
from neutro.utils.conv_utils import (
    from_channels_last,
    shape_to_channels_last,
    to_channels_last,
)


class Conv2D(Layer):
    """2-D convolution layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        )
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding
        self.activation = get_activation(activation)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        self.data_format = data_format

    def build(self, input_shape):
        _, h, w, c = shape_to_channels_last(input_shape, self.data_format)
        kh, kw = self.kernel_size
        self.params['W'] = Tensor(self.kernel_initializer((kh, kw, c, self.filters)))
        self.params['b'] = Tensor(self.bias_initializer((self.filters,)))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        batch, h, w, _ = shape_to_channels_last(input_shape, self.data_format)
        kh, kw = self.kernel_size
        sh, sw = self.strides
        padding = 0
        if self.padding == "same":
            padding = (kh - 1) // 2
        oh = (h + 2 * padding - kh) // sh + 1
        ow = (w + 2 * padding - kw) // sw + 1
        if self.data_format == "channels_first":
            return (batch, self.filters, oh, ow)
        return (batch, oh, ow, self.filters)

    def forward(self, inputs, training=False):
        inputs_nhwc = to_channels_last(inputs, self.data_format)
        out = Conv2DFunction.apply(
            inputs_nhwc, self.params["W"], self.params["b"],
            self.kernel_size, self.strides, self.padding,
        )
        out = from_channels_last(out, self.data_format)
        if self.activation:
            out = self.activation(out)
        return out