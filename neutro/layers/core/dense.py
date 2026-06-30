import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer
from ...activations import get as get_activation
from neutro.autograd import Tensor


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.params['W'] = Tensor(self.kernel_initializer((self.input_dim, self.units)))
        if self.use_bias:
            self.params['b'] = Tensor(self.bias_initializer((self.units,)))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [self.units])

    def forward(self, inputs, training=False):
        z = inputs @ self.params['W']
        if self.use_bias:
            z = z + self.params['b']
        if self.activation:
            return self.activation(z)
        return z