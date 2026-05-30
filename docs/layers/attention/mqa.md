# Multi-Query Attention (MQA)

## What does this layer do?

Multi-Query Attention is a **memory-efficient** variant of Multi-Head Attention. The key insight: all query heads **share** a single key head and a single value head. Instead of H separate K and V projections, you have just one. This dramatically reduces memory usage in the KV cache during generation (by roughly H×) while retaining most of the model quality.

MQA was introduced by Shazeer (2019) and is used in models like PaLM and Falcon.

## How is this different from MHA?

| Feature | MHA | MQA |
|---------|-----|-----|
| Query heads | H | H |
| Key heads | H | **1** |
| Value heads | H | **1** |
| `Wk` shape | `(embed_dim, key_dim)` | `(embed_dim, head_dim)` |
| `Wv` shape | `(embed_dim, key_dim)` | `(embed_dim, head_dim)` |
| KV cache memory | H × full_seq | **1 × full_seq** |

The single key and value heads are **broadcast** to match all H query heads during attention.

## The math, in plain English

**MHA projections:**
$$Q = XW_q,\quad K = XW_k,\quad V = XW_v$$
Where $W_k, W_v \in \mathbb{R}^{D \times (H \cdot d)}$

**MQA projections:**
$$Q = XW_q,\quad K = XW_k,\quad V = XW_v$$
Where $W_k, W_v \in \mathbb{R}^{D \times d}$ — just one head's worth!

The attention computation:
$$K_{\text{broadcast}} = \text{broadcast}(K, \text{heads}=H)$$
$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_{\text{broadcast}}^T}{\sqrt{d}}\right) V_{\text{broadcast}}$$

Every query head uses the **same** K and V, but they each have their own Q projection, so they can still learn to focus on different patterns.

---

## Walking through the code

### File: `neutro/layers/attention/mqa.py`

### Step 1: `__init__` — line 6

```python
class MultiQueryAttention(BaseAttention):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
```

🔍 **Line 6**: Inherits from `BaseAttention` — we get `scaled_dot_product_attention` for free.

🔍 **Line 8**: `num_heads` — number of **query** heads. This is H.

🔍 **Line 10**: `head_dim = key_dim // num_heads` — the dimension of each single head. Since K and V have only one head, their total dimension equals `head_dim` (not `key_dim`).

### Step 2: `build` — line 12

```python
def build(self, input_shape):
    self.embed_dim = input_shape[-1]
    init = get_initializer('glorot_uniform')
    self.params['Wq'], self.params['Wk'], self.params['Wv'] = init((self.embed_dim, self.key_dim)), init((self.embed_dim, self.head_dim)), init((self.embed_dim, self.head_dim))
    self.params['Wo'] = init((self.key_dim, self.embed_dim))
    super().build(input_shape)
```

🔍 **Line 15**: **Here's the key difference from MHA!** 
- `Wq`: `(embed_dim, key_dim)` — same as MHA, maps to all H heads.
- `Wk`: `(embed_dim, head_dim)` — **not** `(embed_dim, key_dim)`! Only one head's worth.
- `Wv`: `(embed_dim, head_dim)` — same as Wk, only one head's worth.

📐 If `key_dim=512`, `num_heads=8`, then `head_dim=64`.
- MHA `Wk`: `(embed_dim, 512)` — 512 parameters per input dim.
- MQA `Wk`: `(embed_dim, 64)` — 64 parameters per input dim. **8× smaller!**

🔍 **Line 16**: `Wo`: `(key_dim, embed_dim)` — the output projection still goes from full `key_dim` back to `embed_dim`. The queries still produce H heads worth of output.

### Step 3: `forward` — line 19

```python
def forward(self, query, value=None, key=None, mask=None, training=False):
    if value is None: value = query
    if key is None: key = value
    batch_size = query.shape[0]
    Q = np.dot(query, self.params['Wq']).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
```

🔍 **Line 23**: **Q projection** is the same as MHA: `(B, S, D) @ (D, key_dim)` → `(B, S, key_dim)` → reshape to `(B, H, S, d)`.

```python
    K, V = np.dot(key, self.params['Wk']).reshape(batch_size, -1, 1, self.head_dim).transpose(0, 2, 1, 3), np.dot(value, self.params['Wv']).reshape(batch_size, -1, 1, self.head_dim).transpose(0, 2, 1, 3)
```

🔍 **Line 24**: **K and V projections** — the key difference!

`np.dot(key, self.params['Wk'])` → `(B, S, head_dim)` — only one head's worth of dimensions.

Then `.reshape(batch_size, -1, 1, head_dim)` — note the `1`! This creates a dummy head dimension of size 1.

Then `.transpose(0, 2, 1, 3)` → `(B, 1, S, d)`.

📐 **MHA K shape**: `(B, H, S, d)`. **MQA K shape**: `(B, 1, S, d)`.

```python
    attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
```

🔍 **Line 25**: Call `scaled_dot_product_attention` from `BaseAttention`. Here's where the magic happens:

- `Q` is `(B, H, S, d)`
- `K` is `(B, 1, S, d)`
- `V` is `(B, 1, S, d)`

NumPy **broadcasts** the `1` in K and V to match `H` during the matrix multiply in `scaled_dot_product_attention`! So effectively, each query head sees the same K and V.

📐 Inside SDPA:
- `Q @ K^T`: `(B, H, S_q, d) @ (B, 1, d, S_kv)` → broadcast → `(B, H, S_q, S_kv)` ✓
- `A @ V`: `(B, H, S_q, S_kv) @ (B, 1, S_kv, d)` → broadcast → `(B, H, S_q, d)` ✓

```python
    out = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.key_dim)
    return np.dot(out, self.params['Wo'])
```

🔍 **Line 26–27**: Merge heads and apply output projection — identical to MHA.

📐 `(B, H, S, d)` → `(B, S, H*d)` = `(B, S, key_dim)` → `(B, S, embed_dim)`.

### Step 4: `backward` — line 29

```python
def backward(self, grad_output):
    # MQA backward is similar to MHA but with summation over heads for K and V
    # Implementing a placeholder for now to focus on structure
    return None
```

🔍 **Line 29**: The MQA backward pass is **currently a placeholder**. A full implementation would follow the same structure as MHA's backward, with one critical difference: gradients for K and V need to be **summed across all H heads** (since the single K/V head was broadcast to all query heads). 

🔍 **Line 32**: Returns `None` — this means MQA currently cannot be used for training. In practice, this is an honest note to the reader that implementing the full backward pass is a good exercise! The structure is:
1. Compute dWo, d_pre_output (same as MHA).
2. Backprop through attention (same as MHA).
3. For dWk and dWv: **sum dK_heads and dV_heads across the head dimension** before computing weight gradients.
4. Sum the input gradients across Q, K, V paths.

## Usage Example

```python
from neutro.layers.attention import MultiQueryAttention
import numpy as np

layer = MultiQueryAttention(num_heads=8, key_dim=512)
x = np.random.randn(32, 10, 256)
layer.build(x.shape)
y = layer(x)  # forward works fine
# layer.backward(grad)  # returns None — training not yet implemented
```

## When to use MQA vs MHA

- **MHA** when you need maximum model quality and have enough memory.
- **MQA** when you're doing long-sequence generation and the KV cache is the bottleneck. The quality loss is often minimal.
- **GQA** (Grouped Query Attention) is a middle ground — see the [GQA guide](gqa.md).

## References

- Shazeer, N. (2019). **Fast Transformer Decoding: One Write-Head is All You Need**. *arXiv:1911.02150*. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
