from ..base import Layer
from ...activations import get as get_activation


class Activation(Layer):
    """Wraps an activation function as a layer."""

    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation = get_activation(activation)

    def forward(self, inputs, training=False):
        return self.activation(inputs)


class ReLU(Activation):
    """ReLU activation layer."""

    def __init__(self, **kwargs):
        super().__init__("relu", **kwargs)


class Softmax(Activation):
    """Softmax activation layer."""

    def __init__(self, **kwargs):
        super().__init__("softmax", **kwargs)


class Sigmoid(Activation):
    """Sigmoid activation layer."""

    def __init__(self, **kwargs):
        super().__init__("sigmoid", **kwargs)


class Tanh(Activation):
    """Tanh activation layer."""

    def __init__(self, **kwargs):
        super().__init__("tanh", **kwargs)