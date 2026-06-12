import numpy as np
from ..base import Layer
from .embedding import Embedding


class TokenPositionEmbedding(Layer):
    """
    Combined token + learnable position embedding.
    
    Internally creates two Embedding sub-layers — one for tokens
    (vocab_size -> dim) and one for positions (max_len -> dim) —
    and adds them in forward().  Both sets of embeddings are
    learnable and get gradients from the backward pass.
    
    This matches the standard Keras/TF pattern:
        token_emb = Embedding(vocab_size, dim)(tokens)
        pos_emb   = Embedding(max_len, dim)(positions)
        x = token_emb + pos_emb
    """
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
        positions = np.arange(seq_len, dtype=np.int32).reshape(1, -1)
        return self.token_emb(inputs) + self.pos_emb(positions)

    def backward(self, grad_output):
        self.token_emb.backward(grad_output)
        # positions had shape (1, seq_len) → sum grad over batch before passing
        self.pos_emb.backward(grad_output.sum(axis=0, keepdims=True))
        return None
