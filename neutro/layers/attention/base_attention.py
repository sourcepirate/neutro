import numpy as np
from ..base import Layer
from neutro.autograd import Tensor, ops as autograd_ops


class BaseAttention(Layer):
    def __init__(self, scale=None, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = q.shape[-1]
        scale = self.scale or np.sqrt(dk)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            scores = scores + Tensor(-1e9) * mask
        attn_weights = autograd_ops.softmax(scores, axis=-1)
        return attn_weights @ v

    @staticmethod
    def create_causal_mask(seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1)
