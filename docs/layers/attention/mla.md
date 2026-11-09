# Multi-Head Latent Attention (MLA)

> **Implementation note:** The current layer relies on the tape-backed
> `Layer.backward`; it does not define `backward`. The manual code below is
> illustrative; the forward math and shapes still apply. See
> [Layer base class](../base.md#backward-the-tape-backed-fallback).


## What does this layer do?

Multi-Head Latent Attention is the attention mechanism behind **DeepSeek-V2 and V3**. Its big idea: instead of caching full Key and Value tensors for the KV cache (which gets enormous for long sequences), MLA compresses them into a **low-rank latent vector** and caches that instead.

Think of it like this: MHA caches the entire encyclopedia. MLA caches a **summary card** and reconstructs the details on the fly.

## The KV cache comparison

| Variant | What gets cached | Cache size per layer per token |
|---------|-----------------|-------------------------------|
| MHA | Full K, V | `2 × H × d` |
| GQA | G groups of K, V | `2 × G × d` |
| MQA | Single K, V | `2 × d` |
| **MLA** | **Compressed latent** | **`kv_latent_dim`** (e.g., 128 vs 2048) |

In DeepSeek-V2, `kv_latent_dim` is dramatically smaller than `H × head_dim` — typically 128–512 vs 2048+.

## The math, in plain English

MLA introduces a **compression-decompression** bottleneck:

### Encoding (compression):
$$c_{kv} = W_{kv}^{\text{down}} x$$

The input $x$ is projected into a low-dimensional **latent vector** $c_{kv}$. This is what gets cached.

### Decoding (decompression):
$$[K; V] = W_{kv}^{\text{up}} c_{kv}$$

The latent is projected back up into the full concatenated Key + Value tensor. Note there's **no RoPE on the content K/V** — DeepSeek handles positional encoding with a separate projection.

### Query side:
$$c_q = W_q^{\text{down}} x, \quad Q = W_q^{\text{up}} c_q$$

The query also goes through compression/decompression, though this doesn't affect the cache size (queries are computed once per token and not cached).

---

## Walking through the code

### File: `neutro/layers/attention/mla.py`

### Step 1: `__init__`

```python
class MultiHeadLatentAttention(Layer):
    def __init__(self, num_heads, head_dim, latent_dim, kv_latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.kv_latent_dim = kv_latent_dim
        self.scale = 1.0 / np.sqrt(head_dim)
```

🔍 MLA inherits directly from `Layer`, NOT from `BaseAttention`. It implements its own attention math inline, but the softmax approach is the same conceptually.

🔍 `latent_dim` — the compressed dimension for **Q** projections.

🔍 `kv_latent_dim` — the compressed dimension for **KV** projections. This is the **key hyperparameter** that determines cache efficiency.

🔍 The scale is pre-computed as `1 / sqrt(head_dim)` — a small optimization over computing `np.sqrt` each time.

### Step 2: `build`

```python
def build(self, input_shape):
    self.embed_dim = input_shape[-1]

    # KV Compression
    self.kv_compress = Dense(self.kv_latent_dim, use_bias=False)
    self.kv_compress.build(input_shape)

    # KV Decompression (to content)
    self.kv_decompress = Dense(self.num_heads * (self.head_dim + self.head_dim))
    self.kv_decompress.build((None, self.kv_latent_dim))
```

🔍 `kv_compress = Dense(kv_latent_dim)` — the **down-projection**: `(embed_dim,)` → `(kv_latent_dim,)`. No bias (standard for efficiency).

🔍 `kv_decompress = Dense(num_heads * (head_dim + head_dim))` — the **up-projection**: `(kv_latent_dim,)` → `(H * 2d,)`. The `2d` is because we output **both** K and V concatenated together.

```python
    # Q projection (to latent)
    self.q_compress = Dense(self.latent_dim, use_bias=False)
    self.q_compress.build(input_shape)

    # Q decompression
    self.q_decompress = Dense(self.num_heads * self.head_dim)
    self.q_decompress.build((None, self.latent_dim))

    # Final projection
    self.wo = Dense(self.embed_dim, use_bias=False)
    self.wo.build((None, self.num_heads * self.head_dim))

    super().build(input_shape)
```

🔍 Same compresion-decompression for Q: `embed_dim → latent_dim → H*d`. The query doesn't affect cache size, but the bottleneck can still help with model quality.

🔍 Output projection: `(H*d,)` → `(embed_dim,)`.

### Step 3: `forward`

```python
def forward(self, x, mask=None, training=False, kv_cache=None, layer_id=None):
    self.x = x
    batch_size, seq_len, _ = x.shape
    H = self.num_heads
    d = self.head_dim

    # 1. Compress & Decompress Q
    q_latent = self.q_compress(x, training=training)
    q = self.q_decompress(q_latent, training=training)
    q = q.reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
```

🔍 **Q pipeline**: compress → decompress → split heads.
- `q_latent`: `(B, S, latent_dim)` — the compressed query.
- `q`: `(B, S, H*d)` — the decompressed full query.
- Final shape: `(B, H, S, d)` — standard multi-head format.

```python
    # 2. Compress & Decompress KV
    kv_latent = self.kv_compress(x, training=training)
```

🔍 **KV compression**: `(B, S, embed_dim)` → `(B, S, kv_latent_dim)`. This **tiny latent** is the magic of MLA.

```python
    # KV caching happens on the LATENT vector in MLA!
    if kv_cache is not None and layer_id is not None:
        # kv_latent is (B, S, D). KVCache expects (B, H, S, d).
        # We treat it as 1 head: (B, 1, S, D)
        kv_latent_reshaped = kv_latent[:, np.newaxis, :, :]
        _, kv_latent_cached = kv_cache.update(kv_latent_reshaped, kv_latent_reshaped, layer_id)
        # Result is (B, 1, S_total, D) -> (B, S_total, D)
        kv_latent = kv_latent_cached[:, 0, :, :]
        seq_len_kv = kv_latent.shape[1]
    else:
        seq_len_kv = seq_len
```

🔍 **KV cache interaction** — the most interesting part!

The standard `KVCache` expects `(B, H, S, d)` shaped tensors. But our latent is `(B, S, kv_latent_dim)`. We **trick** the cache by adding a dummy head dimension: `(B, 1, S, kv_latent_dim)`.

We use `kv_latent_reshaped` for **both** K and V in `kv_cache.update()` (since it's a single latent that represents both). We only keep the cached K output (the first return value is the K cache, which we ignore with `_`).

**Key takeaway**: The cache stores `(B, 1, S_total, kv_latent_dim)` instead of `(B, H, S_total, d)`. If `kv_latent_dim = 128` and `H*d = 2048`, that's **16× less memory** per cached token!

```python
    kv = self.kv_decompress(kv_latent, training=training)
    kv = kv.reshape(batch_size, seq_len_kv, H, 2 * d)
    k = kv[..., :d].transpose(0, 2, 1, 3)
    v = kv[..., d:].transpose(0, 2, 1, 3)
```

🔍 **Decompress** the (cached) latent back to full K and V: `(B, S, kv_latent_dim)` → `(B, S, H*2d)`.

🔍 Split into K and V by slicing the last dimension in half.
- `kv[..., :d]` — first half is K_content.
- `kv[..., d:]` — second half is V_content.
- Both reshaped to `(B, H, S, d)`.

```python
    # 3. Standard Scaled Dot-Product Attention
    scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
    if mask is not None:
        scores += (mask * -1e9)

    attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights /= (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-15)
    self.attn_weights = attn_weights

    out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, H * d)
    return self.wo(out, training=training)
```

🔍 **Standard attention** — identical math to `BaseAttention.scaled_dot_product_attention`. The softmax, scaling, and weighted sum are the same. The innovation of MLA is entirely in **how K and V are produced** (compressed/decompressed), not in how attention is computed.

### Step 4: `backward`

```python
def backward(self, grad_output):
    batch_size, seq_len, _ = self.x.shape
    H = self.num_heads
    d = self.head_dim

    # Backprop through Wo
    grad_wo_in = self.wo.backward(grad_output)
    grad_wo_in = grad_wo_in.reshape(batch_size, seq_len, H, d)
```

🔍 **Output projection backward** — delegates to the `Dense` layer's own `backward`. This is the clean part.

```python
    # Backprop through Attention
    # dV, dWeights, dQ, dK... (omitting details for brevity)

    # Dummy grad for the decompressors to ensure they get updated
    grad_q = self.q_decompress.backward(grad_wo_in.reshape(batch_size, seq_len, -1))
    self.q_compress.backward(grad_q)
```

🔍 **Partial backward for Q path** — backpropagates through `q_decompress` and `q_compress` sub-layers. This is correct but simplified (routes the gradient through the decompressors sequentially).

```python
    # Split grad for KV
    grad_kv = np.random.randn(batch_size, seq_len, H * 2 * d) # Dummy for now
    self.kv_decompress.backward(grad_kv)
    self.kv_compress.backward(grad_kv[:, :, :self.kv_latent_dim]) # Approximate

    return np.random.randn(*self.x.shape) # Return grad_x
```

🔍 **Placeholder for KV path** — uses random noise as gradients. This is **intentionally naive** — MLA's backward requires backpropagating through the attention softmax, then through the decompression, then through the cache-aware latent, which is complex.

In a production MLA (DeepSeek-V2/V3), the backward pass would:
1. Backprop through the attention (same softmax gradient as MHA).
2. Backprop through `kv_decompress` to get `dkv_latent`.
3. Sum `dkv_latent` across timesteps (since cached latents affect all future queries).
4. Backprop through `kv_compress`.

This educational version shows the **structure** (compression → decompression → attention) and proves the forward pass works, while honestly noting the backward is a work in progress.

## Usage Example

```python
from neutro.layers.attention import MultiHeadLatentAttention
import numpy as np

layer = MultiHeadLatentAttention(
    num_heads=16, head_dim=64,
    latent_dim=128, kv_latent_dim=64
)
x = np.random.randn(4, 32, 512)
layer.build(x.shape)
y = layer(x) # (4, 32, 512)
```

For generation with KV cache:
```python
from neutro.layers.attention.kv_cache import KVCache

cache = KVCache()
# First token
y1 = layer(x[:, :1, :], kv_cache=cache, layer_id=0)
# Second token — cache contains compressed latent from step 1
y2 = layer(x[:, 1:2, :], kv_cache=cache, layer_id=0)
```

## References

- DeepSeek-AI. (2024). **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. *arXiv:2405.04434*. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
