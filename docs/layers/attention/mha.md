# Multi-Head Attention (MHA)

Multi-head attention runs scaled dot-product attention `H` times in parallel,
each with its own projection, then concatenates the heads and projects back.

Source: `neutro/layers/attention/mha.py`.

Paper: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).

---

## The math

### 1. Project

```text
Q = X Wq
K = X Wk
V = X Wv
```

For self-attention, `X` is the same tensor for all three. For cross-attention,
`Q` comes from the decoder and `K`, `V` come from the encoder.

### 2. Split into heads

```text
(B, S, key_dim) -> (B, S, H, d) -> (B, H, S, d)
```

`d = key_dim // H`.

### 3. Scaled dot-product attention per head

```text
scores = Q K^T / sqrt(d) # (B, H, S_q, S_kv)
A = softmax(scores, axis=-1)
head = A V # (B, H, S_q, d)
```

### 4. Merge and project

```text
(B, H, S, d) -> (B, S, H*d) = (B, S, key_dim)
output = merged @ Wo # (B, S, embed_dim)
```

---

## `__init__`

```python
class MultiHeadAttention(BaseAttention):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
```

- `num_heads` — `H`, how many parallel heads.
- `key_dim` — total projected width. Must be divisible by `num_heads`.
- `head_dim` — `d = key_dim // num_heads`.

`BaseAttention` supplies `scaled_dot_product_attention` and
`create_causal_mask`.

---

## `build`

```python
def build(self, input_shape):
    self.embed_dim = input_shape[-1]
    init = get_initializer('glorot_uniform')
    self.params['Wq'] = Tensor(init((self.embed_dim, self.key_dim)))
    self.params['Wk'] = Tensor(init((self.embed_dim, self.key_dim)))
    self.params['Wv'] = Tensor(init((self.embed_dim, self.key_dim)))
    self.params['Wo'] = Tensor(init((self.key_dim, self.embed_dim)))
    super.build(input_shape)
```

Shapes:

```text
Wq, Wk, Wv: (embed_dim, key_dim)
Wo: (key_dim, embed_dim)
```

Glorot uniform is the standard transformer initialization.

---

## `_split_heads`

```python
def _split_heads(self, x, batch_size):
    return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
```

Shape walkthrough:

```text
x: (B, S, key_dim)
reshape: (B, S, H, d)
permute: (B, H, S, d)
```

Heads move to axis 1 so the batched matrix multiplies treat each head
independently.

---

## `forward`

```python
def forward(self, query, value=None, key=None, mask=None, training=False,
            kv_cache=None, layer_id=None):
    if value is None:
        value = query
    if key is None:
        key = value

    batch_size = query.shape[0]
    Q = self._split_heads(query @ self.params["Wq"], batch_size)
    K = self._split_heads(key @ self.params["Wk"], batch_size)
    V = self._split_heads(value @ self.params["Wv"], batch_size)

    if kv_cache is not None and layer_id is not None:
        K, V = kv_cache.update(K, V, layer_id)

    attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
    out = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.key_dim)
    return out @ self.params['Wo']
```

Key points:

- `key`/`value` default to `query`, so `layer(x)` is self-attention.
- Pass `layer(decoder_out, memory, memory)` for cross-attention.
- `kv_cache` appends the current keys/values to the cached history along the
  sequence axis. See [KV cache](kv_cache.md).
- `mask` is added to the scores with a large negative number; `1` means
  "blocked."

Shape walkthrough inside `scaled_dot_product_attention`:

```text
Q @ K^T: (B, H, S_q, d) @ (B, H, d, S_kv) -> (B, H, S_q, S_kv)
softmax: (B, H, S_q, S_kv)
A @ V: (B, H, S_q, S_kv) @ (B, H, S_kv, d) -> (B, H, S_q, d)
```

---

## Backward

`MultiHeadAttention` does **not** define `backward`. The base
`Layer.backward` replays `forward` inside a `GradientTape`, so the tape computes:

- `dWq`, `dWk`, `dWv` through each projection and the attention math,
- `dWo` through the output projection,
- `dquery`, `dkey`, `dvalue` for the previous layer.

Conceptually, the reverse pass is:

```text
dWo = merged_heads.T @ d_output
d_merged = d_output @ Wo.T
dA, dV = from d_merged = dA @ V + A @ dV
d_scores = softmax_backward(dA, A)
dQ, dK = d_scores @ K, d_scores.T @ Q
dWq = query.T @ dQ
dWk = key.T @ dK
dWv = value.T @ dV
d_query = dQ @ Wq.T
d_key = dK @ Wk.T
d_value = dV @ Wv.T
```

You do not need to write these by hand for this layer. The derivation is still
useful when reading the tape's result.

> **Softmax gradient:** for `A = softmax(S)`, the vector-Jacobian product is
> `dS = A * (dA - sum(dA * A, axis=-1, keepdims=True))`. The `1/sqrt(d)` factor
> from the scores is applied during the chain back to `Q` and `K`.

---

## Usage

```python
import numpy as np
from neutro.layers import MultiHeadAttention

attn = MultiHeadAttention(num_heads=8, key_dim=512)
x = np.random.randn(2, 10, 512) # (B, S, D)

y = attn(x) # self-attention
print(y.shape) # (2, 10, 512)
```

With a causal mask:

```python
from neutro.layers.attention.base_attention import BaseAttention

mask = BaseAttention.create_causal_mask(seq_len=10)
y = attn(x, mask=mask)
```

`TransformerBlock` builds this mask for you when `causal=True`.

---

## Read next

- [BaseAttention](base_attention.md) — scaled dot-product attention.
- [GQA](gqa.md), [MQA](mqa.md), [MLA](mla.md) — KV-sharing variants.
- [FlashAttention (simplified)](flash_attention.md).
- [TransformerBlock](../transformer/transformer_block.md).
- [Layer base class](../base.md).
