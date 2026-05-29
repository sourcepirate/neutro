from ..base import Layer
from ...activations import get as get_activation

class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation = get_activation(activation)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, grad_output):
        if hasattr(self.activation, 'gradient_fast'):
            return self.activation.gradient_fast(self.inputs, grad_output)
        return grad_output * self.activation.gradient(self.inputs)

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
