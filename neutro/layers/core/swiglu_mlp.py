import numpy as np
from ..base import Layer
from .dense import Dense
from ...activations.silu import SiLU


class SwiGLUMLP(Layer):
    """SwiGLU feed-forward: (SiLU(x @ W1) * (x @ W3)) @ W2.

    Generic MLP used by Llama, Qwen, and as MoE expert.
    Shapes: (B, ..., D) -> gate/value: (B, ..., H) -> (B, ..., D).
    """

    def __init__(self, dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = Dense(hidden_dim)
        self.w2 = Dense(dim)
        self.w3 = Dense(hidden_dim)
        self.silu = SiLU()

    def build(self, input_shape):
        self.w1.build(input_shape)
        self.w2.build((input_shape[0], input_shape[1], self.w1.units) if len(input_shape) == 3 else (*input_shape[:-1], self.w1.units))
        self.w3.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, x, training=False):
        gate = self.w1(x, training)
        activated = self.silu(gate)
        value = self.w3(x, training)
        return self.w2(activated * value, training)
