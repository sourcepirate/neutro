import numpy as np
from .base_attention import BaseAttention
from ...initializers import get as get_initializer

class MultiHeadAttention(BaseAttention):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads

    def build(self, input_shape):
        self.embed_dim = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['Wq'], self.params['Wk'], self.params['Wv'] = init((self.embed_dim, self.key_dim)), init((self.embed_dim, self.key_dim)), init((self.embed_dim, self.key_dim))
        self.params['Wo'] = init((self.key_dim, self.embed_dim))
        super().build(input_shape)

    def _split_heads(self, x, batch_size):
        return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, query, value=None, key=None, mask=None, training=False, kv_cache=None, layer_id=None):
        if value is None: value = query
        if key is None: key = value
        self.query, self.key, self.value, batch_size = query, key, value, query.shape[0]
        self.Q_raw, self.K_raw, self.V_raw = np.dot(query, self.params['Wq']), np.dot(key, self.params['Wk']), np.dot(value, self.params['Wv'])
        Q, K, V = self._split_heads(self.Q_raw, batch_size), self._split_heads(self.K_raw, batch_size), self._split_heads(self.V_raw, batch_size)
        
        if kv_cache is not None and layer_id is not None:
            K, V = kv_cache.update(K, V, layer_id)

        self.attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        out = self.attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.key_dim)
        self.pre_output = out
        return np.dot(out, self.params['Wo'])

    def backward(self, grad_output):
        batch_size, seq_len = grad_output.shape[0], grad_output.shape[1]
        
        # dWo
        pre_output_flat = self.pre_output.reshape(-1, self.key_dim)
        grad_output_flat = grad_output.reshape(-1, self.embed_dim)
        self.grads['Wo'] = pre_output_flat.T @ grad_output_flat
        
        d_pre_output = np.dot(grad_output, self.params['Wo'].T)
        d_attn_output = d_pre_output.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        Q, K, V = self._split_heads(self.Q_raw, batch_size), self._split_heads(self.K_raw, batch_size), self._split_heads(self.V_raw, batch_size)
        
        d_attn_weights, dV_heads = np.matmul(d_attn_output, V.transpose(0, 1, 3, 2)), np.matmul(self.attention_weights.transpose(0, 1, 3, 2), d_attn_output)
        
        d_scores = self.attention_weights * (d_attn_weights - np.sum(d_attn_weights * self.attention_weights, axis=-1, keepdims=True)) / np.sqrt(self.head_dim)
        
        dQ_heads, dK_heads = np.matmul(d_scores, K), np.matmul(d_scores.transpose(0, 1, 3, 2), Q)
        
        dQ_raw = dQ_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.key_dim)
        dK_raw = dK_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.key_dim)
        dV_raw = dV_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.key_dim)
        
        query_flat = self.query.reshape(-1, self.embed_dim)
        key_flat = self.key.reshape(-1, self.embed_dim)
        value_flat = self.value.reshape(-1, self.embed_dim)
        
        self.grads['Wq'] = query_flat.T @ dQ_raw.reshape(-1, self.key_dim)
        self.grads['Wk'] = key_flat.T @ dK_raw.reshape(-1, self.key_dim)
        self.grads['Wv'] = value_flat.T @ dV_raw.reshape(-1, self.key_dim)
        
        return np.dot(dQ_raw, self.params['Wq'].T) + np.dot(dK_raw, self.params['Wk'].T) + np.dot(dV_raw, self.params['Wv'].T)
