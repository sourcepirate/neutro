import numpy as np
from ..base import Layer
from .embedding import Embedding
from neutro.autograd import Tensor


class TokenPositionEmbedding(Layer):
    def __init__(self, vocab_size, max_len, dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(max_len, dim)
        self.max_len = max_len

    def build(self, input_shape):
        self.token_emb.build(input_shape)
        self.pos_emb.build((input_shape[0], self.max_len))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape) + [self.token_emb.output_dim])

    def forward(self, inputs, training=False):
        seq_len = inputs.shape[1]
        positions = Tensor(np.arange(seq_len, dtype=np.int32).reshape(1, -1))
        return self.token_emb(inputs) + self.pos_emb(positions)