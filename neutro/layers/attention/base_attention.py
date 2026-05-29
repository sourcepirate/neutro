import numpy as np
from ..base import Layer

class BaseAttention(Layer):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = q.shape[-1]
        scale = self.scale or np.sqrt(dk)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        if mask is not None: scores += (mask * -1e9)
        self.attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attention_weights /= (np.sum(self.attention_weights, axis=-1, keepdims=True) + 1e-15)
        return np.matmul(self.attention_weights, v)

    @staticmethod
    def create_causal_mask(seq_len):
        """Creates a square causal mask (1 for positions to mask, 0 for allowed)."""
        return np.triu(np.ones((seq_len, seq_len)), k=1)
