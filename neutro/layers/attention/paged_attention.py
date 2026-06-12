import numpy as np
from ..base import Layer
from ...utils.rope_utils import precompute_freqs_cis, apply_rotary_emb


class PagedKVCache:
    """
    Paged KV Cache: manages KV cache in fixed-size blocks (pages).
    
    Physical blocks are pre-allocated in a flat pool.
    A block table per layer maps logical block indices -> physical block IDs.
    Free list tracks which physical blocks are available.
    
    This is the memory management backbone for PagedAttention (vLLM-style).
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
        """
        Store K, V into physical blocks, allocating new blocks as needed.
        
        k, v: (batch, num_heads, seq_len, head_dim)
        
        Layers are independent. Each layer_id gets its own block table.
        """
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
        """Returns (block_table, block_fill) for block-wise attention iteration."""
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
    PagedAttention: Memory-efficient attention with block-level KV cache.
    
    Forward pass uses block-iterated attention with online softmax
    (matching FlashAttention's algorithm but over variable-size blocks
    from the block table). This demonstrates the core vLLM contribution:
    KV cache stored in non-contiguous physical blocks, accessed via
    a block table, with attention computed block by block.
    
    When no PagedKVCache is provided, falls back to standard attention.
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
        self.scale = 1.0 / np.sqrt(self.head_dim)

    def build(self, input_shape):
        self.embed_dim = input_shape[-1]
        self.params['Wq'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
        self.params['Wk'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
        self.params['Wv'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
        self.params['Wo'] = np.random.randn(self.key_dim, self.embed_dim) * 0.02
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _assemble_kv(self, kv_cache, layer_id, batch_size, d):
        """Assemble full K, V from physical blocks for backward."""
        block_table, block_fill = kv_cache.get_block_table(layer_id)
        total = kv_cache.get_num_tokens(layer_id)
        H = self.num_heads

        k_assembled = np.zeros((batch_size, H, total, d))
        v_assembled = np.zeros((batch_size, H, total, d))
        pos = 0
        for phys_id in block_table:
            fill = block_fill[phys_id]
            k_assembled[:, :, pos:pos + fill, :] = \
                kv_cache.kv_blocks[phys_id, 0][np.newaxis, :, :fill, :]
            v_assembled[:, :, pos:pos + fill, :] = \
                kv_cache.kv_blocks[phys_id, 1][np.newaxis, :, :fill, :]
            pos += fill
        return k_assembled, v_assembled

    def _paged_forward(self, Q, kv_cache, layer_id, mask, B, S_q, H, d):
        """
        Block-iterated forward with online softmax.
        
        Iterates over the block table for this layer, loading one physical
        block at a time and merging with running statistics
        (same algorithm as FlashAttention, but tiles = physical blocks).
        """
        block_table, block_fill = kv_cache.get_block_table(layer_id)
        kv_blocks = kv_cache.kv_blocks

        O = np.zeros((B, H, S_q, d))
        L = np.zeros((B, H, S_q, 1))
        M = np.full((B, H, S_q, 1), -np.inf)

        pos = 0
        for phys_id in block_table:
            fill = int(block_fill[phys_id])
            if fill == 0:
                continue

            Kb = kv_blocks[phys_id, 0][np.newaxis, :, :fill, :]
            Vb = kv_blocks[phys_id, 1][np.newaxis, :, :fill, :]
            if B > 1:
                Kb = np.broadcast_to(Kb, (B, H, fill, d))
                Vb = np.broadcast_to(Vb, (B, H, fill, d))

            S = self.scale * (Q @ Kb.transpose(0, 1, 3, 2))
            if mask is not None:
                S -= 1e9 * mask[np.newaxis, np.newaxis, :, pos:pos + fill]

            m_local = np.max(S, axis=-1, keepdims=True)
            P = np.exp(S - m_local)
            l_local = np.sum(P, axis=-1, keepdims=True)

            M_new = np.maximum(M, m_local)
            alpha = np.exp(M - M_new)
            beta = np.exp(m_local - M_new)

            O = alpha * O + beta * (P @ Vb)
            M = M_new
            L = alpha * L + beta * l_local
            pos += fill

        O = O / (L + 1e-15)
        self.paged_M = M
        self.paged_L = L
        return O

    def _standard_forward(self, Q, K, V, mask):
        """Standard softmax attention. Caches attention_weights."""
        S = self.scale * (Q @ K.transpose(0, 1, 3, 2))
        if mask is not None:
            S -= 1e9 * mask
        attn_w = np.exp(S - np.max(S, axis=-1, keepdims=True))
        attn_w /= (np.sum(attn_w, axis=-1, keepdims=True) + 1e-15)
        self.attention_weights = attn_w
        return attn_w @ V

    def forward(self, x, mask=None, training=False, kv_cache=None, layer_id=None):
        self.x = x
        self.mask = mask
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
                elif layer_id in kv_cache.k_cache:
                    total_seq_len += kv_cache.k_cache[layer_id].shape[2]
            self.freqs_cis = precompute_freqs_cis(self.head_dim, total_seq_len)
            if seq_len == 1 and total_seq_len > 1:
                f_cis = self.freqs_cis[total_seq_len - 1:total_seq_len]
            else:
                f_cis = self.freqs_cis[:seq_len]
            Q = apply_rotary_emb(Q, f_cis)
            K = apply_rotary_emb(K, f_cis)

        if kv_cache is not None and layer_id is not None:
            self._cache_used = True
            if isinstance(kv_cache, PagedKVCache):
                kv_cache.update(K, V, layer_id)
                self.K, self.V = self._assemble_kv(kv_cache, layer_id, batch_size, d)
                attn_out = self._paged_forward(Q, kv_cache, layer_id, mask,
                                               batch_size, seq_len, H, d)
            else:
                K, V = kv_cache.update(K, V, layer_id)
                self.K, self.V = K, V
                attn_out = self._standard_forward(Q, K, V, mask)
        else:
            self._cache_used = False
            self.K, self.V = K, V
            attn_out = self._standard_forward(Q, K, V, mask)

        self.Q = Q
        O_merged = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
        self.pre_output = O_merged
        return O_merged @ self.params['Wo']

    def _compute_attention_weights(self, Q, K, V, mask):
        """Recompute attention weights from assembled Q, K, V."""
        S = self.scale * (Q @ K.transpose(0, 1, 3, 2))
        if mask is not None:
            S -= 1e9 * mask
        attn_w = np.exp(S - np.max(S, axis=-1, keepdims=True))
        attn_w /= (np.sum(attn_w, axis=-1, keepdims=True) + 1e-15)
        return attn_w

    def backward(self, grad_output):
        batch_size, seq_len, embed_dim = self.x.shape
        H = self.num_heads
        d = self.head_dim
        K_dim = self.key_dim

        pre_output_flat = self.pre_output.reshape(-1, K_dim)
        grad_output_flat = grad_output.reshape(-1, embed_dim)
        self.grads['Wo'] = pre_output_flat.T @ grad_output_flat

        do_merged = grad_output @ self.params['Wo'].T
        do = do_merged.reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)

        if hasattr(self, 'attention_weights'):
            attn_w = self.attention_weights
        else:
            attn_w = self._compute_attention_weights(
                self.Q, self.K, self.V, self.mask
            )

        d_attn_weights = do @ self.V.transpose(0, 1, 3, 2)
        dV_heads = attn_w.transpose(0, 1, 3, 2) @ do

        d_scores = attn_w * (d_attn_weights - np.sum(
            d_attn_weights * attn_w, axis=-1, keepdims=True
        ))
        d_scores = d_scores / np.sqrt(d)

        dQ_heads = d_scores @ self.K
        dK_heads = d_scores.transpose(0, 1, 3, 2) @ self.Q

        if self._cache_used:
            dK_heads = dK_heads[:, :, -seq_len:, :]
            dV_heads = dV_heads[:, :, -seq_len:, :]

        if self.use_rope:
            dQ_heads = apply_rotary_emb(dQ_heads, np.conj(self.freqs_cis))
            dK_heads = apply_rotary_emb(dK_heads, np.conj(self.freqs_cis))

        dq_flat = dQ_heads.transpose(0, 2, 1, 3).reshape(-1, K_dim)
        dk_flat = dK_heads.transpose(0, 2, 1, 3).reshape(-1, K_dim)
        dv_flat = dV_heads.transpose(0, 2, 1, 3).reshape(-1, K_dim)
        x_flat = self.x.reshape(-1, embed_dim)

        self.grads['Wq'] = x_flat.T @ dq_flat
        self.grads['Wk'] = x_flat.T @ dk_flat
        self.grads['Wv'] = x_flat.T @ dv_flat

        return (dq_flat @ self.params['Wq'].T +
                dk_flat @ self.params['Wk'].T +
                dv_flat @ self.params['Wv'].T).reshape(batch_size, seq_len, embed_dim)
