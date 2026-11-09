# Tutorial 03 — Attention from Scratch

**Time:** about 15 minutes. **Prerequisites:** [Tutorial 01](01-first-model.md)
and [Tutorial 02](02-gradients-and-the-tape.md).

Attention is the core of the transformer. In this lesson we compute it by hand
with NumPy, then use Neutro's `MultiHeadAttention` and a causal mask.

Run the companion script:

```bash
python examples/03_attention_tour.py
```

Expected output, approximately:

```text
Hand-written attention
  x:        (2, 4, 8) (batch, seq, dim)
  Q/K/V:    (2, 4, 8)
  scores:   (2, 4, 4) (batch, query, key)
  weights:  (2, 4, 4)
  output:   (2, 4, 8)
  first query's weights sum to 1.0

Neutro MultiHeadAttention
  output:   (2, 4, 8) Tensor
```

The exact numbers depend on the random seed; the shapes do not.

---

## 1. The idea

Each position asks a question (query `Q`) and compares it with every position's
label (key `K`). The comparison produces a score. Softmax turns the scores into
weights. The output is a weighted sum of the values `V`.

```text
scores  = Q K^T / sqrt(d)     # how much each query matches each key
weights = softmax(scores)     # each query's weights sum to 1
output  = weights V           # weighted blend of values
```

`d` is the per-head dimension. Dividing by `sqrt(d)` keeps the scores from
growing too large as `d` increases.

---

## 2. One head by hand

```python
import numpy as np

rng = np.random.default_rng(0)

batch, seq, dim, head_dim = 2, 4, 8, 4
x = rng.normal(size=(batch, seq, dim))

Wq = rng.normal(size=(dim, dim))
Wk = rng.normal(size=(dim, dim))
Wv = rng.normal(size=(dim, dim))

Q = x @ Wq
K = x @ Wk
V = x @ Wv

scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(head_dim)

def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)

weights = softmax(scores)
output = weights @ V
```

Shape walkthrough:

```text
x:            (B, S, D)      (2, 4, 8)
Q, K, V:      (B, S, D)
K^T:          (B, D, S)
scores:       (B, S, S)      (2, 4, 4)
weights:      (B, S, S)
output:       (B, S, D)      (2, 4, 8)
```

Check that each query's weights sum to 1:

```python
print(weights[0, 0].sum())   # 1.0
```

That is the whole mechanism. The rest is how to split it into heads, stack it
in blocks, and make it efficient.

---

## 3. The same thing with `MultiHeadAttention`

```python
from neutro.layers import MultiHeadAttention

attention = MultiHeadAttention(num_heads=2, key_dim=8)
y = attention(x)

print(y.shape)        # (2, 4, 8)
print(type(y).__name__)  # Tensor
```

Inside the layer:

1. `Wq`, `Wk`, `Wv` project `(B, S, D)` to `(B, S, key_dim)`.
2. The result is split into `H` heads: `(B, S, key_dim) -> (B, H, S, d)`.
3. Each head does the same `scores -> softmax -> weighted sum`.
4. The heads are concatenated: `(B, H, S, d) -> (B, S, key_dim)`.
5. `Wo` projects back to `(B, S, D)`.

For `num_heads=2` and `key_dim=8`, each head has `d = 4`.

See [Multi-Head Attention](../layers/attention/mha.md) for the full code.

---

## 4. Causal masking

A language model predicting the next token must not look at the future. A
causal mask blocks positions above the diagonal:

```python
from neutro.layers.attention.base_attention import BaseAttention

mask = BaseAttention.create_causal_mask(seq)
print(mask)
```

Output:

```text
[[0. 1. 1. 1.]
 [0. 0. 1. 1.]
 [0. 0. 0. 1.]
 [0. 0. 0. 0.]]
```

A `1` means "blocked". The layer adds a very negative number at those
positions, so softmax sends their weight to zero.

```python
y_causal = attention(x, mask=mask)
print(y_causal.shape)   # (2, 4, 8)
```

A useful sanity check: the first token can only attend to itself, so its
causal output should equal the output of running the layer on just that token:

```python
first_token_only = attention(x[:, :1, :])
print(np.allclose(y_causal.data[:, 0], first_token_only.data[:, 0]))
# True
```

That is exactly what the mask promises.

---

## 5. What the layer does not do

`MultiHeadAttention` in this repository is the straightforward implementation:
it materializes the full `(B, H, S, S)` score matrix.

- `FlashAttention` keeps the same API but is **simplified**; it also uses the
  full matrix. See [Implementation status](../IMPLEMENTATION_STATUS.md#attention).
- `GQA`, `MQA`, and `MLA` change how many key/value heads exist, not the core
  `scores -> softmax -> V` step.
- KV caching avoids recomputing past keys/values during generation. See
  [KV cache](../layers/attention/kv_cache.md).

---

## 6. Your turn

1. Change `num_heads` to `4` with `key_dim=8`. What is `head_dim` now?
2. Why must `key_dim` be divisible by `num_heads`?
3. Remove the `/ np.sqrt(head_dim)` scaling. What happens to the softmax
   weights as the score magnitude grows?
4. Build a `(1, 6, 8)` input and a `6 x 6` causal mask. Which token can attend
   to position 0?
5. Compare the causal output for token 0 with a single-token forward pass.
   Does the equality still hold?

<details>
<summary>Answers</summary>

1. `head_dim = 8 / 4 = 2`.
2. The split reshapes the last axis into `(H, d)`. If `key_dim` is not
   divisible by `H`, the reshape cannot preserve the total width.
3. Without scaling, large scores make softmax nearly one-hot: one weight
   approaches 1 and the rest approach 0, so gradients shrink.
4. Only token 0 can attend to position 0; positions 1–5 are blocked from it.
5. Yes. Causal token 0 sees only itself, so it matches a one-token forward pass.

</details>

---

## What to read next

- [Multi-Head Attention](../layers/attention/mha.md)
- [TransformerBlock](../layers/transformer/transformer_block.md)
- [FlashAttention (simplified)](../layers/attention/flash_attention.md)
- [KV cache](../layers/attention/kv_cache.md)
- [Tutorial 04 — Build a tiny transformer LM](04-tiny-transformer-lm.md) *(next)*
