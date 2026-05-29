import numpy as np
from .base_attention import BaseAttention
from ...initializers import get as get_initializer

class MultiQueryAttention(BaseAttention):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads

    def build(self, input_shape):
        self.embed_dim = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['Wq'], self.params['Wk'], self.params['Wv'] = init((self.embed_dim, self.key_dim)), init((self.embed_dim, self.head_dim)), init((self.embed_dim, self.head_dim))
        self.params['Wo'] = init((self.key_dim, self.embed_dim))
        super().build(input_shape)

    def forward(self, query, value=None, key=None, mask=None, training=False):
        if value is None: value = query
        if key is None: key = value
        batch_size = query.shape[0]
        Q = np.dot(query, self.params['Wq']).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K, V = np.dot(key, self.params['Wk']).reshape(batch_size, -1, 1, self.head_dim).transpose(0, 2, 1, 3), np.dot(value, self.params['Wv']).reshape(batch_size, -1, 1, self.head_dim).transpose(0, 2, 1, 3)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        out = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.key_dim)
        return np.dot(out, self.params['Wo'])

    def backward(self, grad_output):
        # MQA backward is similar to MHA but with summation over heads for K and V
        # Implementing a placeholder for now to focus on structure
        return None
