# Multi-Head Attention (MHA)

## What does this layer do?

Multi-Head Attention runs the attention mechanism **H times in parallel**, each with its own set of projections. Instead of one attention computation on the full `embed_dim`, we split into `num_heads` smaller subspaces (`head_dim = key_dim / num_heads`). Each head can learn to focus on different types of relationships — position, syntax, semantics — and the results are concatenated and projected back.

This is the classic "Attention is All You Need" mechanism from Vaswani et al. (2017).

## The math, in plain English

The full MHA operation breaks into three phases:

**Phase 1 — Projection:**

$$Q = XW_q, \quad K = XW_k, \quad V = XW_v$$

Each input position $x_i$ is projected into query, key, and value spaces.

**Phase 2 — Scaled Dot-Product Attention (per head):**

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

Each head $i$ sees a smaller slice: $Q_i$ has shape `(B, S, d)` where $d = \text{head\_dim}$.

**Phase 3 — Concatenation + Output Projection:**

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_o$$

The heads are concatenated back to `(B, S, key_dim)`, then projected back to `(B, S, embed_dim)`.

### Gradient flow

The backward pass computes gradients for all **four** weight matrices ($W_q, W_k, W_v, W_o$). The key steps:
- **$dW_o$**: gradient from the output layer, using the pre-output (the concatenated heads before $W_o$).
- **$dW_q, dW_k, dW_v$**: backprop through attention, then through the projections.
- The **softmax gradient** is: `attention_weights * (d_attn_weights - sum(d_attn_weights * attention_weights))`, scaled by `1 / sqrt(d)`.

---

## Walking through the code

### File: `neutro/layers/attention/mha.py`

### Step 1: `__init__` — line 6

```python
class MultiHeadAttention(BaseAttention):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
```

🔍 **Line 6**: MHA inherits from `BaseAttention`, which gives us `scaled_dot_product_attention` and `create_causal_mask` for free.

🔍 **Line 8**: `num_heads` — how many parallel attention heads to use. Typical values: 8, 12, 16.

🔍 **Line 9**: `key_dim` — the **total** dimension of the projected Q (and K, V) before splitting into heads. Must be divisible by `num_heads`.

🔍 **Line 10**: `head_dim = key_dim // num_heads` — each head operates on this many dimensions. For example, if `key_dim=512` and `num_heads=8`, then `head_dim=64`. The head dimension is typically 64–128 in practice.

### Step 2: `build` — line 12

```python
def build(self, input_shape):
    self.embed_dim = input_shape[-1]
    init = get_initializer('glorot_uniform')
    self.params['Wq'], self.params['Wk'], self.params['Wv'] = init((self.embed_dim, self.key_dim)), init((self.embed_dim, self.key_dim)), init((self.embed_dim, self.key_dim))
    self.params['Wo'] = init((self.key_dim, self.embed_dim))
    super().build(input_shape)
```

🔍 **Line 13**: `embed_dim` is read from the input shape — the dimension of each input token vector.

🔍 **Line 14**: We use Glorot uniform initialization (Xavier), which is standard for Transformer weights.

🔍 **Line 15**: Three weight matrices, each of shape `(embed_dim, key_dim)`:
- $W_q$: maps input → query space
- $W_k$: maps input → key space
- $W_v$: maps input → value space

Why `(embed_dim, key_dim)`? Because the input has `embed_dim` features and we want to produce `key_dim` features (which will then be split into heads).

🔍 **Line 16**: $W_o$ maps the concatenated heads `(key_dim,)` back to `(embed_dim,)`. This is the **output projection**.

### Step 3: `_split_heads` — line 19

```python
def _split_heads(self, x, batch_size):
    return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
```

📐 **Shape walkthrough**: `(B, S, key_dim)` → `(B, S, H, d)` → `(B, H, S, d)`
- Reshape: split the last dimension `key_dim` into `H` groups of `d` each.
- Transpose: swap axes 1 and 2 so the head dimension comes second. This puts the heads in the "batch-like" position so that matrix multiplication over the sequence dimension works correctly.

### Step 4: `forward` — line 22

```python
def forward(self, query, value=None, key=None, mask=None, training=False, kv_cache=None, layer_id=None):
    if value is None: value = query
    if key is None: key = value
```

🔍 **Line 22**: MHA accepts three inputs (`query`, `key`, `value`) but commonly all three are the same tensor (self-attention). The `key` and `value` default to `query` so you can just call `layer(x)` for self-attention.

For cross-attention in encoder-decoder models, you'd pass different tensors: `layer(decoder_output, memory, memory)`.

```python
    self.query, self.key, self.value, batch_size = query, key, value, query.shape[0]
    self.Q_raw, self.K_raw, self.V_raw = np.dot(query, self.params['Wq']), np.dot(key, self.params['Wk']), np.dot(value, self.params['Wv'])
    Q, K, V = self._split_heads(self.Q_raw, batch_size), self._split_heads(self.K_raw, batch_size), self._split_heads(self.V_raw, batch_size)
```

🔍 **Line 25**: We **cache** `query`, `key`, `value` as `self.query`, `self.key`, `self.value` — these are needed in the backward pass to compute weight gradients.

🔍 **Line 26**: **Projection step**: compute Q, K, V by matrix-multiplying the inputs with the learned weights.

📐 **Shape walkthrough for Q**:
- `query` is `(B, S, embed_dim)`
- `Wq` is `(embed_dim, key_dim)`
- Result `self.Q_raw` is `(B, S, key_dim)`

🔍 **Line 27**: Split all three into heads using `_split_heads`.

📐 `Q`: `(B, S, key_dim)` → `(B, H, S, d)`

```python
    if kv_cache is not None and layer_id is not None:
        K, V = kv_cache.update(K, V, layer_id)
```

🔍 **Lines 29–30**: If a KV cache is provided, we **update** it with the current K and V tokens and get back the **full** K, V including all previous tokens. During generation, the cache grows on the sequence dimension so we don't recompute past keys and values. See the [KVCache guide](kv_cache.md) for details.

```python
    self.attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
```

🔍 **Line 32**: Call the inherited `scaled_dot_product_attention` from `BaseAttention`. This computes:
$$ \text{softmax}(QK^T / \sqrt{d}) V $$

and caches the attention weights in `self.attention_weights`.

📐 **Inside SDPA**:
- `Q @ K^T`: `(B, H, S_q, d) @ (B, H, d, S_kv)` → `(B, H, S_q, S_kv)`
- softmax → still `(B, H, S_q, S_kv)`
- result @ V: `(B, H, S_q, S_kv) @ (B, H, S_kv, d)` → `(B, H, S_q, d)`

```python
    out = self.attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.key_dim)
    self.pre_output = out
    return np.dot(out, self.params['Wo'])
```

🔍 **Line 33**: **Merge heads**: transpose back from `(B, H, S, d)` to `(B, S, H, d)` then reshape to `(B, S, key_dim)`. This is the inverse of `_split_heads`.

📐 `(B, H, S, d)` → `(B, S, H, d)` → `(B, S, key_dim)`

🔍 **Line 34**: Cache `self.pre_output` — the merged heads before the output projection. This is needed in the backward pass.

🔍 **Line 35**: **Output projection**: `(B, S, key_dim) @ (key_dim, embed_dim)` → `(B, S, embed_dim)`. This mixes information across the heads back into the original embedding space.

### Step 5: `backward` — line 37

```python
def backward(self, grad_output):
    batch_size, seq_len = grad_output.shape[0], grad_output.shape[1]
```

🔍 **Line 37**: `grad_output` is the gradient of the loss with respect to the output of this layer. Shape is `(B, S, embed_dim)`.

#### dWo — gradient for output projection

```python
    pre_output_flat = self.pre_output.reshape(-1, self.key_dim)
    grad_output_flat = grad_output.reshape(-1, self.embed_dim)
    self.grads['Wo'] = pre_output_flat.T @ grad_output_flat
```

🔍 **Lines 41–43**: Gradient for $W_o$ uses the cached `self.pre_output` (the concatenated heads before projection).

📐 `pre_output_flat`: `(B*S, key_dim)`, `grad_output_flat`: `(B*S, embed_dim)` → `dW_o`: `(key_dim, embed_dim)`

#### Backprop through output projection

```python
    d_pre_output = np.dot(grad_output, self.params['Wo'].T)
    d_attn_output = d_pre_output.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
```

🔍 **Line 45**: `d_pre_output` = `grad_output @ Wo^T` — the gradient flows backward through $W_o$.

🔍 **Line 46**: Reshape and transpose to get back to `(B, H, S, d)` — the multi-head format.

#### Backprop through attention

```python
    Q, K, V = self._split_heads(self.Q_raw, batch_size), self._split_heads(self.K_raw, batch_size), self._split_heads(self.V_raw, batch_size)
    
    d_attn_weights, dV_heads = np.matmul(d_attn_output, V.transpose(0, 1, 3, 2)), np.matmul(self.attention_weights.transpose(0, 1, 3, 2), d_attn_output)
```

🔍 **Line 48**: **Recompute** Q, K, V from the cached raw projections. We don't store the split versions, so we split them again here.

🔍 **Line 50**: Two gradients from the attention output `O = A @ V` (where A = attention_weights):
- Gradient w.r.t. attention weights: $dA = dO @ V^T$
- Gradient w.r.t. V: $dV = A^T @ dO$

```python
    d_scores = self.attention_weights * (d_attn_weights - np.sum(d_attn_weights * self.attention_weights, axis=-1, keepdims=True)) / np.sqrt(self.head_dim)
```

🔍 **Line 52**: The **softmax gradient**. For softmax $y = \text{softmax}(x)$, the gradient is:
$$dy/dx = y \cdot (\delta_{ij} - y_j)$$

In practice: `d_scores = A * (dA - sum(dA * A)) / sqrt(d)`. The division by `sqrt(d)` is because `scores = QK^T / sqrt(d)`.

```python
    dQ_heads, dK_heads = np.matmul(d_scores, K), np.matmul(d_scores.transpose(0, 1, 3, 2), Q)
```

🔍 **Line 54**: Gradients w.r.t. Q and K: $dQ = dScores @ K$ and $dK = dScores^T @ Q$.

#### Backprop through projections

```python
    dQ_raw = dQ_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.key_dim)
    dK_raw = dK_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.key_dim)
    dV_raw = dV_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.key_dim)
```

🔍 **Lines 56–58**: **Merge heads** on the gradients — transpose `(B, H, S, d)` → `(B, S, H, d)` → `(B, S, key_dim)`.

```python
    query_flat = self.query.reshape(-1, self.embed_dim)
    key_flat = self.key.reshape(-1, self.embed_dim)
    value_flat = self.value.reshape(-1, self.embed_dim)
    
    self.grads['Wq'] = query_flat.T @ dQ_raw.reshape(-1, self.key_dim)
    self.grads['Wk'] = key_flat.T @ dK_raw.reshape(-1, self.key_dim)
    self.grads['Wv'] = value_flat.T @ dV_raw.reshape(-1, self.key_dim)
```

🔍 **Lines 60–62**: Flatten the cached inputs to `(B*S, embed_dim)`.

🔍 **Lines 64–66**: Compute weight gradients: `dW = input^T @ grad`. This is the standard formula for a linear layer's weight gradient.

📐 `dWq`: `(embed_dim, B*S) @ (B*S, key_dim)` → `(embed_dim, key_dim)` ✓

```python
    return np.dot(dQ_raw, self.params['Wq'].T) + np.dot(dK_raw, self.params['Wk'].T) + np.dot(dV_raw, self.params['Wv'].T)
```

🔍 **Line 68**: Return the gradient w.r.t. the input. There are **three** paths (Q, K, V), so we sum all three contributions. Each path is `grad @ W^T` — the standard backward of a linear layer.

📐 `(B, S, key_dim) @ (key_dim, embed_dim)` → `(B, S, embed_dim)` for each path, then summed.

## Usage Example

```python
from neutro.layers.attention import MultiHeadAttention
import numpy as np

layer = MultiHeadAttention(num_heads=8, key_dim=512)
x = np.random.randn(32, 10, 256)  # (batch, seq_len, embed_dim)
layer.build(x.shape)
y = layer(x)                        # forward, shape (32, 10, 256)
grad = np.random.randn(32, 10, 256)
dx = layer.backward(grad)           # gradient w.r.t. x
```

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. *NeurIPS*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
