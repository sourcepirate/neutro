# KVCache

`KVCache` stores the **Key** and **Value** tensors from previous timesteps
during autoregressive generation. Instead of recomputing attention over the
whole sequence at every step, we compute Q, K, V for only the new token and
combine them with the cached history.

Source: `neutro/layers/attention/kv_cache.py`.

---

## Why it exists

Without a cache, generating an `N`-token sequence costs
`N + (N-1) + (N-2) + ... = N^2/2` attention work. With a cache, each step
processes one new token, so the total is `O(N)`.

At step `t`:

```text
K_cache = [K_0, ..., K_{t-1}]      # already computed
K_t     = x_t W_k                  # only the new token
K_full  = concat(K_cache, K_t)     # along the sequence axis
scores  = Q_t K_full^T / sqrt(d)   # attends to every past position
```

---

## `__init__`

```python
class KVCache:
    def __init__(self):
        self.k_cache = {}  # layer_id -> K tensor
        self.v_cache = {}  # layer_id -> V tensor
```

`KVCache` is **not** a `Layer`; it has no parameters and no forward/backward.
It is a container keyed by `layer_id`, because each transformer layer needs its
own K/V history.

---

## `update`

```python
def update(self, k, v, layer_id):
    if layer_id not in self.k_cache:
        self.k_cache[layer_id] = k
        self.v_cache[layer_id] = v
    else:
        self.k_cache[layer_id] = self._concat(self.k_cache[layer_id], k)
        self.v_cache[layer_id] = self._concat(self.v_cache[layer_id], v)

    return self.k_cache[layer_id], self.v_cache[layer_id]

@staticmethod
def _concat(previous, new):
    from neutro.autograd import Tensor, ops as autograd_ops

    if isinstance(previous, Tensor) or isinstance(new, Tensor):
        return autograd_ops.concatenate([previous, new], axis=2)
    return np.concatenate([previous, new], axis=2)
```

- `k`, `v` have shape `(B, H, new_tokens, d)`; during generation,
  `new_tokens` is usually 1.
- The first call stores K/V directly.
- Later calls concatenate along **axis 2**, the sequence axis.
- `_concat` uses `autograd_ops.concatenate` for `Tensor` inputs. This matters:
  a plain `np.concatenate` would convert the tensor to a NumPy array and break
  the tape.

Shape growth:

```text
Step 1: (B, H, 1, d)
Step 2: (B, H, 2, d)
Step 3: (B, H, 3, d)
...
Step t: (B, H, t, d)
```

Memory per token: `2 * B * H * d` floats (K and V). For `B=1, H=32, d=128`,
that is 8,192 floats per token, or about 32 KB at 4 bytes per float.

---

## `reset`

```python
def reset(self):
    self.k_cache = {}
    self.v_cache = {}
```

Call this when starting a new sequence or conversation.

---

## Generation loop

```python
from neutro.layers.attention import MultiHeadAttention
from neutro.layers.attention.kv_cache import KVCache
import numpy as np

layer = MultiHeadAttention(num_heads=8, key_dim=512)
layer.build((None, None, 256))
cache = KVCache()

# Simulate 5 autoregressive steps.
seq = [np.random.randn(1, 1, 256) for _ in range(5)]
for token in seq:
    out = layer(token, kv_cache=cache, layer_id=0)

print(cache.k_cache[0].shape)   # (1, 8, 5, 64)
cache.reset()
```

The first call stores the full prompt; later calls append one token and return
the full history. Attention then computes `softmax(Q K^T / sqrt(d)) V` over
that history.

---

## What MLA does differently

`MultiHeadLatentAttention` caches the **compressed latent** vector instead of
full K and V. See [MLA](mla.md). The cache API is the same; only the cached
tensor's shape differs.

---

## Read next

- [MultiHeadAttention](mha.md)
- [MLA](mla.md)
- [TransformerBlock](../transformer/transformer_block.md)
- [GradientTape](../../autograd/tape.md)
