import numpy as np

class Layer:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.trainable = True
        self.built = False
        self.params = {}
        self.grads = {}
        self.input_shape = kwargs.get('input_shape')
        self.output_shape = None

    def build(self, input_shape):
        self.input_shape = input_shape
        self.built = True

    def forward(self, inputs, training=False):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def __call__(self, inputs, *args, **kwargs):
        if not self.built:
            self.build(inputs.shape)
        return self.forward(inputs, *args, **kwargs)

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads
