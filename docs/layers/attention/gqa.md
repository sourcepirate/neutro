# Grouped Query Attention (GQA)

## What does this layer do?

Grouped Query Attention is the **middle ground** between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). Instead of H key/value heads (MHA) or just 1 (MQA), GQA uses **G groups** of key/value heads, where `1 < G < H`. Each group of query heads shares one key/value head.

GQA was introduced by Ainslie et al. (2023) and is used in models like Llama 2 (70B), Mistral, and Gemma.

## The spectrum of attention variants

```
MHA:  H  key heads, H  value heads   ← maximum quality, most memory
GQA:  G  key heads, G  value heads   ← balanced (sweet spot)
MQA:  1  key head,  1  value head    ← maximum efficiency, least memory
```

Where `G` divides `H`, and `heads_per_group = H / G`.

## The math, in plain English

**MHA projections:**
$$Q = XW_q,\quad K = XW_k,\quad V = XW_v$$
Where $W_k \in \mathbb{R}^{D \times (H \cdot d)}$ — H heads of K.

**GQA projections:**
$$Q = XW_q,\quad K = XW_k,\quad V = XW_v$$
Where $W_k \in \mathbb{R}^{D \times (G \cdot d)}$ — only G heads of K.

Then, before attention, we **repeat** K and V to match H query heads:

$$K_{\text{expanded}} = \text{repeat}(K, \text{groups} \to \text{heads})$$
$$V_{\text{expanded}} = \text{repeat}(V, \text{groups} \to \text{heads})$$

Each group of `heads_per_group` query heads shares one K/V head.

---

## Walking through the code

### File: `neutro/layers/attention/gqa.py`

### Step 1: `__init__` — line 6

```python
class GroupedQueryAttention(BaseAttention):
    def __init__(self, num_heads, num_groups, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
        self.heads_per_group = num_heads // num_groups
```

🔍 **Line 6**: Inherits from `BaseAttention` — we get the shared `scaled_dot_product_attention` method.

🔍 **Line 9**: `num_groups` — the number of **key/value groups** (G). This is the knob you turn to trade off quality vs. efficiency.

🔍 **Line 12**: `heads_per_group = num_heads // num_groups` — how many query heads share one K/V head. For example, if `num_heads=32` and `num_groups=8`, then `heads_per_group=4`.

### Step 2: `build` — line 14

```python
def build(self, input_shape):
    self.embed_dim = input_shape[-1]
    init = get_initializer('glorot_uniform')
    self.params['Wq'], self.params['Wk'], self.params['Wv'] = init((self.embed_dim, self.key_dim)), init((self.embed_dim, self.num_groups * self.head_dim)), init((self.embed_dim, self.num_groups * self.head_dim))
    self.params['Wo'] = init((self.key_dim, self.embed_dim))
    super().build(input_shape)
```

🔍 **Line 17**: **Compare the weight shapes across the three variants:**

| Variant | `Wk` shape | `Wv` shape |
|---------|-----------|-----------|
| MHA | `(D, H·d)` | `(D, H·d)` |
| GQA | `(D, G·d)` | `(D, G·d)` |
| MQA | `(D, d)` | `(D, d)` |

Notice that GQA sits right between MHA and MQA: `G·d` where `1 < G < H`.

🔍 **Line 18**: `Wo` is `(key_dim, embed_dim)` — same as always. The output still goes from full `key_dim` back to `embed_dim`.

### Step 3: `forward` — line 21

```python
def forward(self, query, value=None, key=None, mask=None, training=False):
    if value is None: value = query
    if key is None: key = value
    batch_size = query.shape[0]
    Q = np.dot(query, self.params['Wq']).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
```

🔍 **Line 25**: **Q projection** — same as MHA. Q still produces H heads.

📐 `(B, S, D) @ (D, key_dim)` → `(B, S, H·d)` → reshaped to `(B, H, S, d)`

```python
    K, V = np.dot(key, self.params['Wk']).reshape(batch_size, -1, self.num_groups, self.head_dim).transpose(0, 2, 1, 3), np.dot(value, self.params['Wv']).reshape(batch_size, -1, self.num_groups, self.head_dim).transpose(0, 2, 1, 3)
```

🔍 **Line 26**: **K and V projections** — note `num_groups` in the reshape, not `num_heads`!

📐 K shape: `(B, S, G·d)` → reshape to `(B, S, G, d)` → transpose to `(B, G, S, d)`

```python
    K, V = np.repeat(K, self.heads_per_group, axis=1), np.repeat(V, self.heads_per_group, axis=1)
```

🔍 **Line 27**: **This is the core GQA operation!** `np.repeat` along axis=1 (the head/group dimension) copies each group `heads_per_group` times.

📐 **Shape transformation**:
- Before repeat: K is `(B, G, S, d)` — G groups.
- After repeat: K is `(B, H, S, d)` — each group repeated 4× to match H heads.
- `np.repeat(K, 4, axis=1)` means: `[group1, group2]` → `[group1, group1, group1, group1, group2, group2, group2, group2]`

**Important**: `np.repeat` copies each group **contiguously**, so queries in the same group attend to the same K/V. This is different from `np.tile` which would interleave them.

```python
    attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
```

🔍 **Line 28**: Now both Q and K have shape `(B, H, S, d)` — SDPA works identically to MHA. The only difference is that K/V heads within a group are identical.

```python
    out = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.key_dim)
    return np.dot(out, self.params['Wo'])
```

🔍 **Lines 29–30**: Merge heads and output projection — identical to MHA.

### Step 4: `backward` — line 32

```python
def backward(self, grad_output):
    return None
```

🔍 **Line 32**: The GQA backward is also a **placeholder**. A full implementation would:
1. Follow MHA's backward for the attention math (dWo, d_attn, dQ, dK, dV).
2. **Sum dK and dV across each group** (all query heads in a group share the same K/V, so their gradients must be summed).
3. Use `heads_per_group` to determine which dK slices to sum.

This is intentionally left as an exercise — it's a great way to test your understanding of both MHA backward and the group structure!

## Visual summary: how GQA's groups work

```
Query heads:    [Q0, Q1, Q2, Q3 | Q4, Q5, Q6, Q7 | Q8, Q9, Q10, Q11 | ...]
                     |    |    |      |    |    |      |    |     |    |
K/V groups:         [   K0, V0   |    K1, V1     |     K2, V2      | ...]
                                                     
heads_per_group = 4, num_groups = num_heads / 4
```

Each group of 4 query heads shares one K/V head. The `np.repeat` call is what makes this sharing happen.

## Usage Example

```python
from neutro.layers.attention import GroupedQueryAttention
import numpy as np

# 32 query heads, 8 groups → 4 query heads per group
layer = GroupedQueryAttention(num_heads=32, num_groups=8, key_dim=2048)
x = np.random.randn(16, 20, 512)
layer.build(x.shape)
y = layer(x)  # forward works, shape (16, 20, 512)
```

## References

- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**. *EMNLP 2023*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
