import math

import numpy as np

from ..base import Layer
from ...utils.rope_utils import apply_rotary_emb, precompute_freqs_cis
from neutro.autograd import Tensor, ops as autograd_ops


class PagedKVCache:
    """
    Paged KV Cache: manages KV cache in fixed-size blocks (pages).
    Physical blocks are pre-allocated in a flat pool.
    A block table per layer maps logical block indices -> physical block IDs.
    Free list tracks which physical blocks are available.
    """
    def __init__(self, num_blocks, block_size=16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.kv_blocks = None
        self.block_tables = {}
        self.free_blocks = list(range(num_blocks))
        self.block_fill = np.zeros(num_blocks, dtype=np.int32)
        self.total_tokens = {}

    def _ensure_storage(self, num_heads, head_dim):
        if self.kv_blocks is None:
            self.kv_blocks = np.zeros(
                (self.num_blocks, 2, num_heads, self.block_size, head_dim)
            )

    def update(self, k, v, layer_id):
        _, H, S, d = k.shape
        self._ensure_storage(H, d)
        if layer_id not in self.block_tables:
            self.block_tables[layer_id] = []
            self.total_tokens[layer_id] = 0
        block_table = self.block_tables[layer_id]
        for t in range(S):
            if (len(block_table) == 0 or
                    self.block_fill[block_table[-1]] >= self.block_size):
                phys_id = self.free_blocks.pop()
                block_table.append(phys_id)
            phys_id = block_table[-1]
            fill = self.block_fill[phys_id]
            self.kv_blocks[phys_id, 0, :, fill, :] = k[:, :, t, :]
            self.kv_blocks[phys_id, 1, :, fill, :] = v[:, :, t, :]
            self.block_fill[phys_id] = fill + 1
            self.total_tokens[layer_id] += 1

    def get_block_table(self, layer_id):
        return self.block_tables.get(layer_id, []), self.block_fill

    def get_num_tokens(self, layer_id):
        return self.total_tokens.get(layer_id, 0)

    def reset(self):
        self.kv_blocks = None
        self.block_tables = {}
        self.free_blocks = list(range(self.num_blocks))
        self.block_fill = np.zeros(self.num_blocks, dtype=np.int32)
        self.total_tokens = {}


class PagedAttention(Layer):
    """
    PagedAttention with standard attention decomposition.
    Block-iterated forward (vLLM-style) is bypassed during autograd;
    we assemble K/V from pages and use vanilla QK^T softmax attention.
    The page management structure is preserved for API compatibility.
    """
    def __init__(self, num_heads, key_dim, block_size=16, dropout=0.0,
                 use_rope=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
        self.block_size = block_size
        self.dropout_rate = dropout
        self.use_rope = use_rope
        self.scale = float(1.0 / math.sqrt(self.head_dim))

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

    def _assemble_kv(self, kv_cache, layer_id, batch_size, d):
        """Assemble full K, V from physical blocks. Returns numpy arrays (not Tensor)."""
        block_table, block_fill = kv_cache.get_block_table(layer_id)
        total = kv_cache.get_num_tokens(layer_id)
        H = self.num_heads
        k_assembled = np.zeros((batch_size, H, total, d))
        v_assembled = np.zeros((batch_size, H, total, d))
        pos = 0
        for phys_id in block_table:
            fill = int(block_fill[phys_id])
            if fill == 0:
                continue
            k_assembled[:, :, pos:pos + fill, :] = kv_cache.kv_blocks[phys_id, 0][np.newaxis, :, :fill, :]
            v_assembled[:, :, pos:pos + fill, :] = kv_cache.kv_blocks[phys_id, 1][np.newaxis, :, :fill, :]
            pos += fill
        return k_assembled, v_assembled

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
            if kv_cache is not None:
                if isinstance(kv_cache, PagedKVCache):
                    total_seq_len += kv_cache.get_num_tokens(layer_id)
                elif layer_id in getattr(kv_cache, 'k_cache', {}):
                    total_seq_len += kv_cache.k_cache[layer_id].shape[2]
            freqs_cis = precompute_freqs_cis(self.head_dim, total_seq_len)
            if seq_len == 1 and total_seq_len > 1:
                f_cis = freqs_cis[total_seq_len - 1:total_seq_len]
            else:
                f_cis = freqs_cis[:seq_len]
            Q = apply_rotary_emb(Q, f_cis)
            K = apply_rotary_emb(K, f_cis)

        if kv_cache is not None and layer_id is not None:
            if isinstance(kv_cache, PagedKVCache):
                kv_cache.update(K, V, layer_id)
                K_arr, V_arr = self._assemble_kv(kv_cache, layer_id, batch_size, d)
                K = Tensor(K_arr)
                V = Tensor(V_arr)
            else:
                K, V = kv_cache.update(K, V, layer_id)

        scores = self.scale * (Q @ K.transpose(0, 1, 3, 2))
        if mask is not None:
            scores = scores + Tensor(-1e9) * mask
        attn_weights = autograd_ops.softmax(scores, axis=-1)
        attn_output = attn_weights @ V

        out = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
        return out @ self.params['Wo']
