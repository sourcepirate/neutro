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
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding
        self.activation = get_activation(activation)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

    def build(self, input_shape):
        # input_shape: (batch, height, width, channels)
        _, h, w, c = input_shape
        kh, kw = self.kernel_size
        # W shape: (kh, kw, c, f) -> for im2col we'll want (f, c, kh, kw)
        self.params['W'] = self.kernel_initializer((kh, kw, c, self.filters))
        self.params['b'] = self.bias_initializer((self.filters,))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides
        padding = 0
        if self.padding == 'same':
            padding = (kh - 1) // 2
        oh = (h + 2*padding - kh) // sh + 1
        ow = (w + 2*padding - kw) // sw + 1
        return (batch, oh, ow, self.filters)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        batch, h, w, c = inputs.shape
        kh, kw, _, f = self.params['W'].shape
        sh, sw = self.strides

        # Internally use (N, C, H, W)
        x = inputs.transpose(0, 3, 1, 2)
        # Weights (kh, kw, c, f) -> (f, c, kh, kw)
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
            return self.activation(out)
        return out

    def backward(self, grad_output):
        if self.activation:
            if hasattr(self.activation, 'gradient_fast'):
                grad_output = self.activation.gradient_fast(self.z, grad_output)
            else:
                grad_output = grad_output * self.activation.gradient(self.z)
                
        # grad_output shape: (batch, oh, ow, f)
        batch, oh, ow, f = grad_output.shape
        kh, kw, c, _ = self.params['W'].shape
        sh, sw = self.strides
        
        # Transpose grad_output for easier computation
        dout = grad_output.transpose(3, 1, 2, 0).reshape(f, -1)
        
        # db
        self.grads['b'] = np.sum(grad_output, axis=(0, 1, 2))
        
        # dW
        dW = dout @ self.x_cols.T
        self.grads['W'] = dW.reshape(f, c, kh, kw).transpose(2, 3, 1, 0)
        
        # dX
        W = self.params['W'].transpose(3, 2, 0, 1)
        dx_cols = W.reshape(f, -1).T @ dout
        
        padding = 0
        if self.padding == 'same':
            padding = (kh - 1) // 2
            
        dx = col2im_indices(dx_cols, (batch, c, self.input_shape[1], self.input_shape[2]), kh, kw, padding=padding, stride=sh)
        return dx.transpose(0, 2, 3, 1)
