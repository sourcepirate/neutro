import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer
from ...activations import get as get_activation
from ...utils.conv_utils import im2col_indices, col2im_indices

class Conv2D(Layer):
    """
    2D Convolution layer (e.g. spatial convolution over images).
    
    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width.
        padding: One of "valid" or "same" (case-insensitive).
        activation: Activation function to use.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', data_format='channels_last', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding
        self.activation = get_activation(activation)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
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

    def build(self, input_shape):
        _, h, w, c = self._shape_to_channels_last(input_shape)
        kh, kw = self.kernel_size
        self.params['W'] = self.kernel_initializer((kh, kw, c, self.filters))
        self.params['b'] = self.bias_initializer((self.filters,))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        batch, h, w, _ = self._shape_to_channels_last(input_shape)
        kh, kw = self.kernel_size
        sh, sw = self.strides
        padding = 0
        if self.padding == 'same':
            padding = (kh - 1) // 2
        oh = (h + 2*padding - kh) // sh + 1
        ow = (w + 2*padding - kw) // sw + 1
        if self.data_format == 'channels_first':
            return (batch, self.filters, oh, ow)
        return (batch, oh, ow, self.filters)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        inputs_nhwc = self._to_channels_last(inputs)
        batch, h, w, c = inputs_nhwc.shape
        kh, kw, _, f = self.params['W'].shape
        sh, sw = self.strides

        x = inputs_nhwc.transpose(0, 3, 1, 2)
        W = self.params['W'].transpose(3, 2, 0, 1)
        b = self.params['b'].reshape(-1, 1)

        padding = 0
        if self.padding == 'same':
            padding = (kh - 1) // 2

        self.x_cols = im2col_indices(x, kh, kw, padding=padding, stride=sh)
        res = W.reshape(f, -1) @ self.x_cols + b
        
        oh = (h + 2*padding - kh) // sh + 1
        ow = (w + 2*padding - kw) // sw + 1
        
        out = res.reshape(f, oh, ow, batch).transpose(3, 1, 2, 0)
        self.z = out

        if self.activation:
            out = self.activation(out)
        return self._from_channels_last(out)

    def backward(self, grad_output):
        grad_output_nhwc = self._to_channels_last(grad_output)
        if self.activation:
            if hasattr(self.activation, 'gradient_fast'):
                grad_output_nhwc = self.activation.gradient_fast(self.z, grad_output_nhwc)
            else:
                grad_output_nhwc = grad_output_nhwc * self.activation.gradient(self.z)
                
        batch, oh, ow, f = grad_output_nhwc.shape
        kh, kw, c, _ = self.params['W'].shape
        sh, sw = self.strides
        
        dout = grad_output_nhwc.transpose(3, 1, 2, 0).reshape(f, -1)
        
        self.grads['b'] = np.sum(grad_output_nhwc, axis=(0, 1, 2))
        
        dW = dout @ self.x_cols.T
        self.grads['W'] = dW.reshape(f, c, kh, kw).transpose(2, 3, 1, 0)
        
        W = self.params['W'].transpose(3, 2, 0, 1)
        dx_cols = W.reshape(f, -1).T @ dout
        
        padding = 0
        if self.padding == 'same':
            padding = (kh - 1) // 2
            
        _, h, w, _ = self._shape_to_channels_last(self.input_shape)
        dx = col2im_indices(dx_cols, (batch, c, h, w), kh, kw, padding=padding, stride=sh)
        return self._from_channels_last(dx.transpose(0, 2, 3, 1))
