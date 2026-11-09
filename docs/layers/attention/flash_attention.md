# FlashAttention (Simplified Implementation)

> **Status: simplified.** This layer keeps the FlashAttention-shaped API
> (projections, head splitting, optional RoPE, optional KV cache, output
> projection) but computes **standard** `QK^T -> softmax -> V`. The tiled
> online-softmax algorithm from Dao et al. is **not** implemented.
>
> If you are looking for the exact kernel described in the paper, this is not
> it. If you want an attention layer that slots into a transformer and is easy
> to read, this is it. See [Implementation status](../../IMPLEMENTATION_STATUS.md#attention).

Source: `neutro/layers/attention/flash_attention.py`.

---

## What the real FlashAttention does

Standard attention materializes the whole score matrix:

```text
S = Q K^T / sqrt(d) # (B, H, S, S)
P = softmax(S, axis=-1)
O = P V # (B, H, S, d)
```

For a long sequence, `S` is the memory problem: `S x S` floats per head.

FlashAttention avoids materializing `S` by processing Q, K, and V in tiles and
maintaining running softmax statistics (a running max and a running sum). The
output is exact, but the intermediate matrix never exists in full.

This repository does **not** implement that tiling. The `block_size_r` and
`block_size_c` arguments are accepted for API compatibility but are not used in
the forward pass.

---

## What this implementation does

```python
def forward(self, x, mask=None, training=False, kv_cache=None, layer_id=None):
    batch_size, seq_len, _ = x.shape
    H = self.num_heads
    d = self.head_dim
    K_dim = self.key_dim

    Q = (x @ self.params['Wq']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
    K = (x @ self.params['Wk']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
    V = (x @ self.params['Wv']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)

    # optional RoPE, optional KV cache
    scores = self.scale * (Q @ K.transpose(0, 1, 3, 2))
    if mask is not None:
        scores = scores + Tensor(-1e9) * mask
    attn_weights = autograd_ops.softmax(scores, axis=-1)
    attn_output = attn_weights @ V

    out = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
    return out @ self.params['Wo']
```

Shape walkthrough:

```text
x: (B, S, D)
Q, K, V: (B, S, key_dim) before splitting
after reshape: (B, S, H, d)
after transpose:(B, H, S, d)
scores: (B, H, S, S)
softmax: (B, H, S, S)
attn_output: (B, H, S, d)
concat: (B, S, H*d) = (B, S, key_dim)
output: (B, S, D) after Wo
```

`self.scale = 1 / sqrt(head_dim)`, the usual scaled dot-product factor.

---

## The pieces

### `__init__`

```python
def __init__(self, num_heads, key_dim, block_size_r=64, block_size_c=64,
             dropout=0.0, use_rope=False, **kwargs):
```

- `num_heads` — number of attention heads, `H`.
- `key_dim` — total projected width; `head_dim = key_dim // num_heads`.
- `block_size_r`, `block_size_c` — accepted for API compatibility; unused.
- `dropout` — stored rate; not applied in this forward.
- `use_rope` — apply rotary position embeddings to Q and K.

### `build`

Creates four projection matrices:

```text
Wq, Wk, Wv: (D, key_dim)
Wo: (key_dim, D)
```

Initialization is small normal values (`randn * 0.02`), a common transformer
choice.

### RoPE

When `use_rope=True`:

```python
freqs_cis = precompute_freqs_cis(self.head_dim, total_seq_len)
Q = apply_rotary_emb(Q, f_cis)
K = apply_rotary_emb(K, f_cis)
```

During generation (`seq_len == 1`), the frequency slice is chosen from the end
of the precomputed table so the new token gets the correct position.

### KV cache

```python
if kv_cache is not None and layer_id is not None:
    K, V = kv_cache.update(K, V, layer_id)
```

The cache concatenates new keys/values along the sequence axis and returns the
full history. This is an inference-time optimization; see
[KV cache](kv_cache.md).

### Masking

```python
scores = scores + Tensor(-1e9) * mask
```

A `1` in the mask becomes a very negative score, so softmax sends it to zero.
`TransformerBlock` builds the causal mask.

---

## What is missing compared with the paper

- No tiled Q/K/V traversal.
- No online softmax (running max/sum).
- No fused kernel; the full `(B, H, S, S)` score matrix is materialized.
- Dropout is not applied in this forward.

For a tiny CPU model, none of that matters. For long sequences, it does. The
docstring in the source says the same thing explicitly.

---

## If you want to implement the real algorithm

A useful learning exercise:

1. Split Q, K, V into row/column tiles.
2. Track `m` (running max), `l` (running sum), and `O` (running output) per
   query row.
3. For each tile, compute local scores, update `m` and `l`, rescale the old
   output, and accumulate `P @ V`.
4. Normalize `O` by `l` at the end.

The math is in [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135).
Until then, treat this layer as "attention with FlashAttention's interface."

---

## Read next

- [BaseAttention](base_attention.md) — scaled dot-product attention.
- [MHA](mha.md) — the reference multi-head implementation.
- [KV cache](kv_cache.md) — autoregressive decoding.
- [Implementation status](../../IMPLEMENTATION_STATUS.md) — the honest feature table.
