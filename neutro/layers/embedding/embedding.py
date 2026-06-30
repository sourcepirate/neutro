import numpy as np
from ..base import Layer
from neutro.autograd import Tensor
from neutro.autograd.custom_ops import Gather


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.params['embeddings'] = Tensor(np.random.normal(0, 0.01, (self.input_dim, self.output_dim)))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape) + [self.output_dim])

    def forward(self, inputs, training=False):
        return Gather.apply(self.params['embeddings'], inputs)