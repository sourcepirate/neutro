import numpy as np
from ..base import Layer
from ...utils.rope_utils import precompute_freqs_cis, apply_rotary_emb
from neutro.autograd import Tensor, ops as autograd_ops


class FlashAttention(Layer):
    """
    FlashAttention decomposition using standard attention ops.
    The tiling algorithm from Dao et al. (2022) is bypassed during autograd;
    we use vanilla QK^T softmax attention instead. The name and
    parameter structure are preserved for API compatibility.
    """
    def __init__(self, num_heads, key_dim, block_size_r=64, block_size_c=64, dropout=0.0, use_rope=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
        self.block_size_r = block_size_r
        self.block_size_c = block_size_c
        self.dropout_rate = dropout
        self.use_rope = use_rope
        self.scale = float(1.0 / np.sqrt(self.head_dim))

    def build(self, input_shape):
        self.embed_dim = input_shape[-1]
        init = np.random.randn
        self.params['Wq'] = Tensor(init(self.embed_dim, self.key_dim) * 0.02)
        self.params['Wk'] = Tensor(init(self.embed_dim, self.key_dim) * 0.02)
        self.params['Wv'] = Tensor(init(self.embed_dim, self.key_dim) * 0.02)
        self.params['Wo'] = Tensor(init(self.key_dim, self.embed_dim) * 0.02)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, x, mask=None, training=False, kv_cache=None, layer_id=None):
        batch_size, seq_len, _ = x.shape
        H = self.num_heads
        d = self.head_dim
        K_dim = self.key_dim

        Q = (x @ self.params['Wq']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
        K = (x @ self.params['Wk']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
        V = (x @ self.params['Wv']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)

        if self.use_rope:
            total_seq_len = seq_len
            if kv_cache and layer_id in getattr(kv_cache, 'k_cache', {}):
                total_seq_len += kv_cache.k_cache[layer_id].shape[2]
            freqs_cis = precompute_freqs_cis(self.head_dim, total_seq_len)
            if seq_len == 1 and total_seq_len > 1:
                f_cis = freqs_cis[total_seq_len - 1:total_seq_len]
            else:
                f_cis = freqs_cis[:seq_len]
            Q = apply_rotary_emb(Q, f_cis)
            K = apply_rotary_emb(K, f_cis)

        if kv_cache is not None and layer_id is not None:
            K, V = kv_cache.update(K, V, layer_id)

        scores = self.scale * (Q @ K.transpose(0, 1, 3, 2))
        if mask is not None:
            scores = scores + Tensor(-1e9) * mask
        attn_weights = autograd_ops.softmax(scores, axis=-1)
        attn_output = attn_weights @ V

        out = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
        return out @ self.params['Wo']
