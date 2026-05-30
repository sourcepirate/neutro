# BaseAttention

## What does this do?

`BaseAttention` is the foundation for all attention layers in neutro. It provides two shared utilities that every attention variant needs: the **scaled dot-product attention** computation and a helper for **causal masking**. You never use `BaseAttention` by itself — it's the parent class for `MultiHeadAttention`, `MultiQueryAttention`, `GroupedQueryAttention`, and others.

## The math, in plain English

Scaled dot-product attention is the core equation of the Transformer revolution:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**What does this mean?**

- **$Q$ (Query)**: "What am I looking for?" — one vector per input position.
- **$K$ (Key)**: "What do I contain?" — one vector per input position.
- **$V$ (Value)**: "What information should I pass along?" — one vector per input position.
- **$QK^T$**: Dot products between every query and every key. A large dot product means "this query matches this key."
- **$\sqrt{d_k}$**: Scaling factor. Without it, large dot products push softmax into regions with tiny gradients.
- **softmax**: Converts scores into a probability distribution (they sum to 1.0 per query position).
- **softmax(...) $\times V$**: Weighted sum of the values — you get back mostly the values whose keys matched your query.

### Causal masking

For autoregressive models (language models that predict the next token), position $i$ should only attend to positions $j \leq i$. The causal mask is an **upper-triangular matrix** filled with 1s above the diagonal:

```
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```

A `1` means "mask this position out." We add a large negative number (`-1e9`) to masked positions before softmax, so their attention weight becomes ~0.

---

## Walking through the code

### File: `neutro/layers/attention/base_attention.py`

### `__init__` — line 5

```python
class BaseAttention(Layer):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale
```

🔍 **Line 5**: `BaseAttention` inherits from `Layer`, neutro's base class for all layers. Every attention variant will inherit from this class.

🔍 **Line 6**: `scale` is an **optional** fixed scaling factor. By default it's `None`, which means the actual scale will be computed as `1 / sqrt(d_k)` at runtime. You might set a custom scale if you want temperature-like control over attention sharpness.

### `scaled_dot_product_attention` — line 9

```python
def scaled_dot_product_attention(self, q, k, v, mask=None):
    dk = q.shape[-1]
    scale = self.scale or np.sqrt(dk)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / scale
```

🔍 **Line 9**: Parameters `q, k, v` are already in **multi-head format**: shape `(batch, heads, seq_len, head_dim)`. The caller (MHA, MQA, etc.) is responsible for splitting heads before calling this method.

🔍 **Line 10**: `dk = q.shape[-1]` — the dimension of each individual head. This is `head_dim`, NOT the full `key_dim`.

🔍 **Line 11**: `scale = self.scale or np.sqrt(dk)` — if no custom scale was set, we use `sqrt(d_k)`. This is the standard Transformer scaling factor.

🔍 **Line 12**: The **attention scores** = $Q K^T / \sqrt{d_k}$. 

`k.transpose(0, 1, 3, 2)` swaps the last two axes of K so the matrix multiply works:
- `q` is `(B, H, S_q, d)`
- `k` after transpose is `(B, H, d, S_kv)`
- Result `scores` is `(B, H, S_q, S_kv)` — every query vs every key

📐 **Shape walkthrough**: `(B, H, S_q, d) @ (B, H, d, S_kv)` → `(B, H, S_q, S_kv)`

```python
    if mask is not None:
        scores += (mask * -1e9)
```

🔍 **Line 13**: Apply the mask. `mask` has a `1` where positions should be masked out. Multiplying by `-1e9` (a very large negative number) means those positions will have near-zero probability after softmax. We use `+=` so existing scores are preserved for unmasked positions.

```python
    self.attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    self.attention_weights /= (np.sum(self.attention_weights, axis=-1, keepdims=True) + 1e-15)
    return np.matmul(self.attention_weights, v)
```

🔍 **Line 14**: **Numerically stable softmax**. The raw approach $\frac{e^{s_i}}{\sum e^{s_j}}$ can overflow if scores are large. By subtracting the max score for each row first (`scores - max`), all exponents are ≤ 0, guaranteeing numerical stability.

This is a **cached** value — `self.attention_weights` is stored on the object because the backward pass needs it. You'll see this used in `MultiHeadAttention.backward()`.

🔍 **Line 15**: Divide by the sum of exponents (plus a tiny `1e-15` epsilon to prevent division by zero). The result is a probability distribution: each row sums to 1.0.

🔍 **Line 16**: The **weighted sum of values**: `(B, H, S_q, S_kv) @ (B, H, S_kv, d)` → `(B, H, S_q, d)`. Each query position now holds a blend of all value vectors, weighted by how much they "matched" that query.

📐 **Shape walkthrough**: `(B, H, S_q, S_kv) @ (B, H, S_kv, d)` → `(B, H, S_q, d)`

### `create_causal_mask` — line 18

```python
@staticmethod
def create_causal_mask(seq_len):
    """Creates a square causal mask (1 for positions to mask, 0 for allowed)."""
    return np.triu(np.ones((seq_len, seq_len)), k=1)
```

🔍 **Line 19**: This is a `@staticmethod` — it doesn't need `self` because it's a pure function of `seq_len`.

🔍 **Line 21**: `np.triu(..., k=1)` gives the upper triangle **above** the main diagonal:
- `np.ones((seq_len, seq_len))` — a square of 1s
- `np.triu(..., k=1)` — zeros out the diagonal and below
- Result: `mask[i, j] = 1` if `j > i` (future positions are masked)

This 2D mask gets broadcast across the batch and heads dimensions during the forward pass.

## How subclasses use this

Every attention layer (MHA, MQA, GQA) follows this pattern:

1. **Project** the input to Q, K, V via learned weight matrices.
2. **Split** the projections into `(batch, heads, seq_len, head_dim)` format.
3. **(Optional) Interact with KV cache** to support autoregressive generation.
4. Call `self.scaled_dot_product_attention(Q, K, V, mask)` — **this method**.
5. **Merge** the heads back and project through the output weight `Wo`.

The beauty of this design is that the complex softmax + masking logic lives in one place, and each subclass only needs to implement its specific projection and head-splitting strategy.
