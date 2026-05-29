import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer
from ...utils.conv_utils import im2col_indices, col2im_indices

class Conv1D(Layer):
    """
    1D Convolution layer (e.g. temporal convolution).

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 1 integer, specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of 1 integer, specifying the stride of the convolution.
        padding: One of "valid" or "same" (case-insensitive).
        activation: Activation function to use.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
    """
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides,)
        self.padding = padding
        self.activation = activation 
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

    def build(self, input_shape):
        # input_shape: (batch, steps, channels)
        _, steps, c = input_shape
        k = self.kernel_size[0]
        self.params['W'] = self.kernel_initializer((k, c, self.filters))
        self.params['b'] = self.bias_initializer((self.filters,))
        super().build(input_shape)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        batch, steps, c = inputs.shape
        k = self.kernel_size[0]
        s = self.strides[0]
        f = self.filters

        # Reshape to 2D for im2col: (batch, steps, 1, c)
        x = inputs[:, :, None, :].transpose(0, 3, 1, 2)
        W = self.params['W'][:, None, :, :].transpose(3, 2, 0, 1)
        b = self.params['b'].reshape(-1, 1)

        padding = 0
        if self.padding == 'same':
            padding = (k - 1) // 2

        self.x_cols = im2col_indices(x, k, 1, padding=(padding, 0), stride=(s, 1))
        res = W.reshape(f, -1) @ self.x_cols + b
        
        out_steps = (steps + 2*padding - k) // s + 1
        
        out = res.reshape(f, out_steps, 1, batch).transpose(3, 1, 2, 0).squeeze(2)
        self.z = out
        return out

    def backward(self, grad_output):
        # grad_output shape: (batch, out_steps, f)
        batch, out_steps, f = grad_output.shape
        k, c, _ = self.params['W'].shape
        s = self.strides[0]
        
        # Reshape grad_output to (batch, out_steps, 1, f)
        dout_4d = grad_output[:, :, None, :]
        dout = dout_4d.transpose(3, 1, 2, 0).reshape(f, -1)
        
        self.grads['b'] = np.sum(grad_output, axis=(0, 1))
        
        dW = dout @ self.x_cols.T
        self.grads['W'] = dW.reshape(f, c, k, 1).transpose(2, 3, 1, 0).squeeze(1)
        
        W = self.params['W'][:, None, :, :].transpose(3, 2, 0, 1)
        dx_cols = W.reshape(f, -1).T @ dout
        
        padding = 0
        if self.padding == 'same':
            padding = (k - 1) // 2
            
        dx = col2im_indices(dx_cols, (batch, c, self.input_shape[1], 1), k, 1, padding=(padding, 0), stride=(s, 1))
        return dx.transpose(0, 2, 3, 1).squeeze(2)
