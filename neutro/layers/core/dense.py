import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer
from ...activations import get as get_activation

class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.params['W'] = self.kernel_initializer((self.input_dim, self.units))
        if self.use_bias:
            self.params['b'] = self.bias_initializer((self.units,))
        super().build(input_shape)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.z = np.dot(inputs, self.params['W'])
        if self.use_bias:
            self.z += self.params['b']
        
        if self.activation:
            return self.activation(self.z)
        return self.z

    def backward(self, grad_output):
        if self.activation:
            if hasattr(self.activation, 'gradient_fast'):
                grad_output = self.activation.gradient_fast(self.z, grad_output)
            else:
                grad_output = grad_output * self.activation.gradient(self.z)
        
        inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        
        self.grads['W'] = np.dot(inputs_flat.T, grad_output_flat)
        if self.use_bias:
            self.grads['b'] = np.sum(grad_output_flat, axis=0)
        
        return np.dot(grad_output, self.params['W'].T)
