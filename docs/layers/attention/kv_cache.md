# KVCache

## What does this do?

`KVCache` stores the **Key** and **Value** tensors from previous timesteps during autoregressive generation. Instead of recomputing attention over the entire sequence at every step, we compute Q, K, V for just the **new token** and then combine with the cached K, V from all previous tokens.

Without a KV cache, generating a 1024-token sequence would require $O(N^2)$ total attention work ($N + (N-1) + (N-2) + \dots = N^2/2$). With a cache, it's $O(N)$ — each step only processes one new token's worth of Q, K, V.

## The math, in plain English

At generation step $t$:

**With cache:**
- $K_{\text{cache}} = [K_0, K_1, \dots, K_{t-1}]$ — already computed and stored.
- Compute $K_t = x_t W_k$ — only for the new token.
- $K_{\text{full}} = [K_{\text{cache}}, K_t]$ — concatenate along the sequence dimension.
- Attention scores: $Q_t K_{\text{full}}^T / \sqrt{d}$ — attends to ALL positions but only computed the new K.

**Without cache:**
- Would need to recompute $K_0, K_1, \dots, K_{t-1}$ from scratch every step.

---

## Walking through the code

### File: `neutro/layers/attention/kv_cache.py`

### `__init__` — line 11

```python
class KVCache:
    def __init__(self):
        self.k_cache = {}  # layer_id -> k_tensor
        self.v_cache = {}  # layer_id -> v_tensor
```

🔍 **Line 11**: `KVCache` is **not** a `Layer` — it has no parameters, no forward/backward, and doesn't inherit from anything. It's a simple container.

🔍 **Line 12**: `k_cache` is a dictionary mapping `layer_id` (an integer or string identifying which layer in a multi-layer model) to the cached K tensor for that layer.

🔍 **Line 13**: `v_cache` — same thing for V tensors.

**Why key by `layer_id`?** A Transformer has many attention layers (e.g., 32 for Llama 7B). Each layer needs its own K and V cache. The `layer_id` ensures that `update()` on layer 0 only affects layer 0's cache.

### `update` — line 15

```python
def update(self, k, v, layer_id):
    """
    Updates the cache with new K and V, and returns the full history.
    k, v: (batch, num_heads, 1, head_dim) - usually just one token during generation
    """
    if layer_id not in self.k_cache:
        self.k_cache[layer_id] = k
        self.v_cache[layer_id] = v
    else:
        # Concatenate along the sequence dimension (axis 2)
        self.k_cache[layer_id] = np.concatenate([self.k_cache[layer_id], k], axis=2)
        self.v_cache[layer_id] = np.concatenate([self.v_cache[layer_id], v], axis=2)
        
    return self.k_cache[layer_id], self.v_cache[layer_id]
```

🔍 **Line 18**: `k, v` have shape `(B, H, 1, d)` — that's a **single** new token's K and V.

🔍 **Line 20**: **First token**: no previous cache exists, so we just store the new K and V directly.

📐 After first token: `k_cache[layer_id]` is `(B, H, 1, d)` — one token.

🔍 **Line 22**: `k_cache[layer_id]` already contains tokens from previous steps. We **concatenate** along `axis=2` (the sequence dimension).

📐 **Shape evolution during generation:**
- Step 1: store `(B, H, 1, d)` — cache is `(B, H, 1, d)`
- Step 2: new K is `(B, H, 1, d)`, concatenate → cache becomes `(B, H, 2, d)`
- Step 3: cache grows to `(B, H, 3, d)`
- Step $t$: cache is `(B, H, t, d)`

**Memory growth**: Each layer's cache grows by `2 × B × H × d` floats per token. For a 32-layer model with `d=128, H=32, batch=1`, that's `2 × 1 × 32 × 128 = 8,192` floats per token, or ~32 KB per token at 4-byte precision. For 4K tokens → ~128 MB for the full model.

🔍 **Line 28**: Return the **full cache** (including both old and new tokens). The attention layer uses these to compute attention over the full history while only doing the QKV projection for the current token.

### `reset` — line 30

```python
def reset(self):
    self.k_cache = {}
    self.v_cache = {}
```

🔍 **Line 30**: Clears both caches. This is called when you start generating a new sequence (e.g., a new conversation turn).

## Usage Example: Generation loop

```python
from neutro.layers.attention import MultiHeadAttention
from neutro.layers.attention.kv_cache import KVCache
import numpy as np

layer = MultiHeadAttention(num_heads=8, key_dim=512)
layer.build((None, None, 256))
cache = KVCache()

# Simulate autoregressive generation of 5 tokens
seq = [np.random.randn(1, 1, 256) for _ in range(5)]

outputs = []
for i, token in enumerate(seq):
    out = layer(token, kv_cache=cache, layer_id=0)
    outputs.append(out)
    # KVCache grows internally — layer receives full K, V

# After generation:
print(cache.k_cache[0].shape)  # (1, 8, 5, 64) — all 5 tokens cached
# (5 = head_dim = 512/8)

cache.reset()  # Ready for a new sequence
```

## How attention layers use the cache

Here's the interaction pattern (visible in MHA's forward, line 29):

```python
# In the attention layer's forward:
if kv_cache is not None and layer_id is not None:
    K, V = kv_cache.update(K, V, layer_id)
```

- **First call**: K is `(B, H, S, d)` for the full prompt. Cache stores it.
- **Subsequent calls**: K is `(B, H, 1, d)` for one new token. Cache appends it and returns `(B, H, S+1, d)`.
- The attention layer then computes `softmax(Q @ K^T / sqrt(d)) @ V` using the larger K, V, attending to the full history.

## What MLA does differently

In [Multi-Head Latent Attention](mla.md), the cache stores the **compressed latent** $c_{kv}$ instead of the full K and V:

```python
# In MLA forward:
kv_latent_reshaped = kv_latent[:, np.newaxis, :, :]  # (B, 1, S, kv_latent_dim)
_, kv_latent_cached = kv_cache.update(kv_latent_reshaped, kv_latent_reshaped, layer_id)
```

This is the same `KVCache` class — the only difference is **what** gets stored. For MLA, it's a much smaller tensor.

## References

- The KV cache pattern is described in the Transformer decoding literature. Key references include the original Transformer paper (Vaswani et al., 2017) which describes autoregressive decoding, and practical guides like [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
