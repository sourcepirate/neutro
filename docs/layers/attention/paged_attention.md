# PagedAttention

## What does this layer do?

`PagedAttention` implements the **block-level KV cache management** introduced by Kwon et al. (2023) in the vLLM system. It stores Key and Value tensors in **fixed-size blocks (pages)** managed through a **block table**, eliminating memory fragmentation and enabling near-100% cache utilization.

### The memory problem

A naive KV cache (`KVCache`) grows by `np.concatenate` at each generation step:

```
Step 1: cache = [K0]                    # shape (B, H, 1, d)
Step 2: cache = [K0, K1]               # shape (B, H, 2, d)
Step 3: cache = [K0, K1, K2]           # shape (B, H, 3, d)
```

This causes **fragmentation**: each reallocation may copy the entire cache to a new memory region. The cache can only shrink/expand at sequence boundaries.

### The paged solution

`PagedKVCache` pre-allocates a flat pool of physical blocks:

```
Physical blocks (pre-allocated):
┌─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │  ...
│ (4 tok) │ (4 tok) │ (4 tok) │ (4 tok) │
└─────────┴─────────┴─────────┴─────────┘

Block table for layer 0: [0, 1, 2, ...]
                          │  │  └── physical block 2  (logical block 2)
                          │  └───── physical block 1  (logical block 1)
                          └──────── physical block 0  (logical block 0)

Each block fills up to block_size, then a new block is allocated.
No reallocation or concatenation — just write to the next slot.
```

### Block-iterated attention

Instead of assembling a contiguous K, V tensor, `PagedAttention`'s forward pass iterates over the block table, loading one physical block at a time and computing attention with **online softmax** (same algorithm as FlashAttention, but tiles = physical blocks of variable size).

---

## Walking through the code

### File: `neutro/layers/attention/paged_attention.py`

---

## Class 1: `PagedKVCache` — lines 6–73

### `__init__` — line 16

```python
class PagedKVCache:
    def __init__(self, num_blocks, block_size=16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.kv_blocks = None
        self.block_tables = {}
        self.free_blocks = list(range(num_blocks))
        self.block_fill = np.zeros(num_blocks, dtype=np.int32)
        self.total_tokens = {}
```

🔍 **Line 16**: `PagedKVCache` is **not** a `Layer` — it's a container, like `KVCache`.

🔍 **Line 19**: `kv_blocks` is lazily initialized in `_ensure_storage`. Shape: `(num_blocks, 2, num_heads, block_size, head_dim)`. The `2` is for K (index 0) and V (index 1).

🔍 **Line 20**: `block_tables` — dictionary mapping `layer_id` to a list of physical block IDs. This is the core data structure that maps the logical sequence to physical memory.

🔍 **Line 21**: `free_blocks` — list of available physical block IDs. Starts as `[0, 1, 2, ..., num_blocks-1]`. Blocks are popped from the end as they're allocated.

🔍 **Line 22**: `block_fill` — integer array tracking how many valid tokens are stored in each physical block. A block is "full" when `block_fill[phys_id] == block_size`.

🔍 **Line 23**: `total_tokens` — dictionary tracking total cached tokens per layer. Used for RoPE position calculation.

### `_ensure_storage` — line 25

```python
def _ensure_storage(self, num_heads, head_dim):
    if self.kv_blocks is None:
        self.kv_blocks = np.zeros(
            (self.num_blocks, 2, num_heads, self.block_size, head_dim)
        )
```

🔍 **Line 25**: Lazy initialization. We don't know `num_heads` and `head_dim` until the first `update()` call, so the physical storage is allocated on first use.

📐 **Shape**: `(num_blocks, 2, H, block_size, d)` — a flat pool of 5D tensors.

### `update` — line 31

```python
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
```

🔍 **Line 39**: Extract `num_heads`, `seq_len`, `head_dim` from the incoming K tensor.

🔍 **Line 42–43**: First update for this `layer_id`? Create a new block table entry.

🔍 **Line 48**: Process each token in the incoming sequence (this handles both prefill with S>1 and decode with S=1).

🔍 **Lines 49–52**: **Block allocation trigger** — if no blocks exist or the last block is full, pop a free block and append it to the block table. This is the key inefficiency of the naive cache: we never concatenate, we just write to the next available slot.

📐 **Block fill evolution** with `block_size=4`:
- Token 0: block 0, fill=1
- Token 1: block 0, fill=2
- Token 2: block 0, fill=3
- Token 3: block 0, fill=4
- Token 4: block 0 is full → allocate block 1, fill=1
- ...

🔍 **Lines 54–58**: Store K and V for this token into the physical block at position `fill`, then increment `block_fill` and `total_tokens`.

🔍 **Why store K[0] not K?** The physical storage has no batch dimension. We take the first batch element (common for generation). `k[:, :, t, :]` is `(B, H, d)`; for B=1 this squeezes to `(H, d)`.

### `get_block_table` — line 61

```python
def get_block_table(self, layer_id):
    return self.block_tables.get(layer_id, []), self.block_fill
```

🔍 **Line 62**: Returns `(block_table, block_fill)` — a tuple. The block table is a list of physical IDs; `block_fill` is the fill-level array for all blocks. The `PagedAttention._paged_forward` method uses both to iterate over valid KV data.

### `get_num_tokens` — line 65

Returns the total cached tokens for a layer (used by RoPE position computation).

### `reset` — line 68

Clears all state and restores the free list. Called between generation sequences.

---

## Class 2: `PagedAttention` — lines 76–294

### `__init__` — line 88

```python
class PagedAttention(Layer):
    def __init__(self, num_heads, key_dim, block_size=16, dropout=0.0,
                 use_rope=False, **kwargs):
```

🔍 **Line 94**: `block_size` serves double duty: it's both the page size for `PagedKVCache` and the tile size for block-iterated attention.

🔍 **Line 97**: `self.scale = 1.0 / sqrt(head_dim)` — the standard attention scaling factor.

### `build` — line 99

Standard MHA-style weight initialization: four projection matrices `Wq, Wk, Wv, Wo` with shape `(embed_dim, key_dim)` or `(key_dim, embed_dim)`.

### `_assemble_kv` — line 110

```python
def _assemble_kv(self, kv_cache, layer_id, batch_size, d):
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
```

🔍 **Line 110**: Assembles a contiguous K, V tensor from the block-based storage. This is used ONLY for the backward pass and for the regular KVCache fallback — the forward pass (`_paged_forward`) reads blocks directly.

🔍 **Why assemble for backward?** The attention gradient formulas require the full K, V to compute `d_scores` and `d_attn_weights`. We need contiguous tensors for the standard MHA-style backward.

📐 **Iteration**: For each physical block, copy its non-padding tokens (up to `block_fill[phys_id]`) into the assembled tensors.

### `_paged_forward` — line 128

```python
def _paged_forward(self, Q, kv_cache, layer_id, mask, B, S_q, H, d):
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
```

🔍 **Line 128**: This is the heart of PagedAttention. It implements the **online softmax** algorithm (identical to FlashAttention) but iterates over **physical blocks** from the block table instead of uniform tiles.

🔍 **Lines 139–141**: Running statistics: `O` (output), `L` (sum of exponentials), `M` (max score). Same initialization as FlashAttention.

🔍 **Line 143**: `pos` tracks the cumulative token position across blocks — used for mask slicing.

🔍 **Lines 149–153**: Load one physical block's K and V. `kv_blocks[phys_id, 0]` is `(H, block_size, d)`; we slice to `fill` valid tokens and add a batch dimension. For B>1, broadcast the cache (same KV for all batch items).

📐 `(H, fill, d)` → `[np.newaxis]` → `(1, H, fill, d)` → `broadcast_to` → `(B, H, fill, d)`

🔍 **Line 155**: **Block-level attention scores**: `S = scale * Q @ Kb^T`. Shape: `(B, H, S_q, fill)` — only this block's worth of scores.

🔍 **Line 156–157**: Apply mask at the block level: `mask[:, pos:pos+fill]` slices the mask region corresponding to this block's KV positions.

🔍 **Lines 159–161**: **Local softmax**: `m_local` is the max within this block, `P` is the exponentiated scores, `l_local` is the local sum.

🔍 **Lines 163–169**: **Online softmax merge**: 
- `M_new = max(M, m_local)` — new global max
- `alpha = exp(M - M_new)` — rescaling for old statistics
- `beta = exp(m_local - M_new)` — rescaling for new block
- `O = alpha * O + beta * (P @ Vb)` — merge output
- `L = alpha * L + beta * l_local` — merge sum

🔍 **Line 172**: **Final normalization**: `O = O / L`. Same as FlashAttention.

🔍 **Lines 173–174**: Cache `paged_M` and `paged_L` (not currently used in backward — the backward uses assembled K, V and recomputed attention weights).

### `_standard_forward` — line 177

```python
def _standard_forward(self, Q, K, V, mask):
    S = self.scale * (Q @ K.transpose(0, 1, 3, 2))
    if mask is not None:
        S -= 1e9 * mask
    attn_w = np.exp(S - np.max(S, axis=-1, keepdims=True))
    attn_w /= (np.sum(attn_w, axis=-1, keepdims=True) + 1e-15)
    self.attention_weights = attn_w
    return attn_w @ V
```

🔍 **Line 177**: Standard softmax attention. Used when no cache or a regular `KVCache` is provided. Caches `self.attention_weights` for the standard backward path.

### `forward` — line 187

```python
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
```

🔍 **Lines 195–197**: Standard Q, K, V projections and head splitting. Identical to FlashAttention and MHA.

```python
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
```

🔍 **Lines 199–212**: RoPE support. Handles both `PagedKVCache` (via `get_num_tokens`) and regular `KVCache` (via `k_cache[layer_id].shape[2]`). During decode with cache, only the last position's RoPE is applied.

```python
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
```

🔍 **Line 214**: **Cache routing** — three paths:

1. **PagedKVCache** (line 216–220): 
   - `kv_cache.update(K, V, layer_id)` — store current K, V into blocks
   - `_assemble_kv` — build contiguous K, V for backward
   - `_paged_forward` — block-iterated attention with online softmax

2. **Regular KVCache** (line 221–224):
   - Standard `kv_cache.update` (concatenate)
   - Standard softmax attention

3. **No cache** (line 225–228):
   - Self-attention with the current Q, K, V

```python
    self.Q = Q
    O_merged = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
    self.pre_output = O_merged
    return O_merged @ self.params['Wo']
```

🔍 **Lines 230–233**: Store Q for backward, merge heads, apply output projection.

### `_compute_attention_weights` — line 235

Helper that recomputes `softmax(QK^T / sqrt(d))` from the assembled Q, K, V. Used by the backward pass when the paged forward path didn't cache `attention_weights`.

### `backward` — line 244

```python
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
```

🔍 **Lines 250–255**: Standard `dWo` and `dO` computation — same as MHA.

```python
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
```

🔍 **Lines 257–262**: Two paths:
- **Standard path** (`self.attention_weights` exists): use cached weights from `_standard_forward`
- **Paged path** (no `attention_weights`): recompute from assembled `self.K`, `self.V`

🔍 **Lines 264–273**: **Standard MHA gradient computation** — the same softmax gradient formula used in `MultiHeadAttention.backward`:
- `d_attn_weights = dO @ V^T` — gradient of attention logits
- `dV = W^T @ dO` — gradient through weighted sum
- `d_scores = W * (d_attn_weights - sum(W * d_attn_weights)) / sqrt(d)` — softmax gradient
- `dQ = d_scores @ K` — gradient for Q
- `dK = d_scores^T @ Q` — gradient for K

```python
    if self._cache_used:
        dK_heads = dK_heads[:, :, -seq_len:, :]
        dV_heads = dV_heads[:, :, -seq_len:, :]
```

🔍 **Lines 275–277**: **Critical cache-aware fix**: When a cache was used, `dK_heads` and `dV_heads` include gradients for ALL cached tokens. But the current input's K, V projections only need gradients for the CURRENT input's tokens (the last `seq_len` positions). The cached tokens' gradients were already computed and applied when they were processed.

📐 `dK_heads` shape: `(B, H, S_total, d)` → `(B, H, S_current, d)` by slicing `[:, :, -seq_len:, :]`

```python
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
```

🔍 **Lines 279–281**: **Reverse RoPE** — apply complex conjugate rotation to unwrap the gradients.

🔍 **Lines 283–296**: Flatten and compute weight gradients `dW = x^T @ d_proj`, then sum input gradients from all three paths.

---

## Three forward paths summary

| Cache type | Forward algorithm | Backward path | Key behavior |
|---|---|---|---|
| `None` | `_standard_forward` | Uses `attention_weights` cache | Pure self-attention |
| `KVCache` (regular) | `_standard_forward` | Uses `attention_weights` cache | Concatenate + standard attention |
| `PagedKVCache` | `_paged_forward` (online softmax) | Recomputes weights from assembled K,V | Block iteration + online softmax |

## Usage Example

```python
from neutro.layers.attention.paged_attention import PagedAttention, PagedKVCache
import numpy as np

layer = PagedAttention(num_heads=4, key_dim=32)
layer.build((1, 1, 32))

cache = PagedKVCache(num_blocks=256, block_size=16)

# Prefill: process 10 tokens
x_prefill = np.random.randn(1, 10, 32)
out = layer(x_prefill, kv_cache=cache, layer_id=0)
print(cache.get_num_tokens(0))  # 10 tokens cached

# Decode: one token at a time
x_decode = np.random.randn(1, 1, 32)
for _ in range(5):
    out = layer(x_decode, kv_cache=cache, layer_id=0)
    print(out.shape)  # (1, 1, 32)

print(cache.get_num_tokens(0))  # 15 tokens cached
cache.reset()
```

## References

- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C., Gonzalez, J., Zhang, H., & Stoica, I. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention**. *SOSP*. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- The online softmax algorithm is from FlashAttention: Dao et al. (2022), [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
