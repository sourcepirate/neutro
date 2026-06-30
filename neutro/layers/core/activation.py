from ..base import Layer
from ...activations import get as get_activation

class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation = get_activation(activation)

    def forward(self, inputs, training=False):
        return self.activation(inputs)

class ReLU(Activation):
    def __init__(self, **kwargs):
        super().__init__('relu', **kwargs)

class Softmax(Activation):
    def __init__(self, **kwargs):
        super().__init__('softmax', **kwargs)

class Sigmoid(Activation):
    def __init__(self, **kwargs):
        super().__init__('sigmoid', **kwargs)

class Tanh(Activation):
    def __init__(self, **kwargs):
        super().__init__('tanh', **kwargs)