# KV Cache

## Theory

The KV Cache is an optimization for autoregressive generation. During inference, each new token attends to all previous tokens. Without caching, we recompute $K$ and $V$ for every token at every step — an $O(L^2)$ cost per step. The KV Cache stores the key and value projections from previous steps, reducing per-step cost to $O(L)$.

### How it works

At step $t$:
1. Compute $Q_t$ from the current token only (shape: $(1, 1, d)$).
2. Fetch $K_{1:t-1}, V_{1:t-1}$ from cache.
3. Compute $K_t, V_t$ from the current token and **append** to cache.
4. Compute attention: $Q_t \cdot [K_c, K_t]^T$.

## Implementation Guide

### File: `neutro/layers/attention/kv_cache.py`

```python
class KVCache:
    def __init__(self):
        self.k_cache = {}  # {layer_id: ndarray}
        self.v_cache = {}

    def get_or_create(self, layer_id, shape):
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = np.zeros(shape)
            self.v_cache[layer_id] = np.zeros(shape)
        return self.k_cache[layer_id], self.v_cache[layer_id]
```

- `layer_id` distinguishes which layer the cache belongs to (each TransformerBlock has its own cache).
- The cache shape is `(batch, num_heads, seq_len, head_dim)`.
- In `TransformerBlock.forward`, the cache is populated at line 50: cached values are retrieved, and the mask is extended to account for past tokens.

## Usage Example

```python
from neutro.layers.attention.kv_cache import KVCache

cache = KVCache()
model.generate(start_tokens, max_new_tokens=100, temperature=0.8)
# Internally uses KVCache for efficient decoding
```

## References

- Vaswani, A., et al. (2017). **Attention Is All You Need**. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
