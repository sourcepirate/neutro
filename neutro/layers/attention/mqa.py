import numpy as np
from .base_attention import BaseAttention
from ...initializers import get as get_initializer
from neutro.autograd import Tensor, ops as autograd_ops


class MultiQueryAttention(BaseAttention):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads

    def build(self, input_shape):
        self.embed_dim = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['Wq'] = Tensor(init((self.embed_dim, self.key_dim)))
        self.params['Wk'] = Tensor(init((self.embed_dim, self.head_dim)))
        self.params['Wv'] = Tensor(init((self.embed_dim, self.head_dim)))
        self.params['Wo'] = Tensor(init((self.key_dim, self.embed_dim)))
        super().build(input_shape)

    def forward(self, query, value=None, key=None, mask=None, training=False):
        if value is None: value = query
        if key is None: key = value
        if not isinstance(query, Tensor):
            query = Tensor(query)
        if not isinstance(key, Tensor):
            key = Tensor(key)
        if not isinstance(value, Tensor):
            value = Tensor(value)

        batch_size = query.shape[0]
        Q = (query @ self.params['Wq']).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (key @ self.params['Wk']).reshape(batch_size, -1, 1, self.head_dim).transpose(0, 2, 1, 3)
        V = (value @ self.params['Wv']).reshape(batch_size, -1, 1, self.head_dim).transpose(0, 2, 1, 3)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        out = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.key_dim)
        return out @ self.params['Wo']