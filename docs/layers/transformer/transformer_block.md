# TransformerBlock

The transformer block is the repeating unit of GPT, Llama, Qwen, and most
modern language models. It combines self-attention with a feed-forward network,
residual connections, and normalization.

Source: `neutro/layers/transformer/transformer_block.py`.

---

## What it computes

Two variants:

### Pre-Norm (GPT-2, Llama, Qwen)

```text
h = x + Dropout(Attention(LayerNorm(x)))
out = h + Dropout(FFN(LayerNorm(h)))
```

### Post-Norm (original Transformer, BERT)

```text
out1 = LayerNorm(x + Dropout(Attention(x)))
out = LayerNorm(out1 + Dropout(FFN(out1)))
```

Pre-norm keeps activations controlled before each sub-layer and is more stable
for deep stacks. Post-norm is the original paper's ordering.

---

## `__init__`

```python
def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,
             causal=False, use_flash=False, pre_norm=False, **kwargs):
    super().__init__(**kwargs)
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.causal = causal
    self.pre_norm = pre_norm
    if use_flash:
        self.attention = FlashAttention(num_heads=num_heads, key_dim=embed_dim)
    else:
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.att = self.attention # backwards-compatible alias
    self.ffn = [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
```

- `self.attention` — attention. `use_flash=True` selects `FlashAttention`, which
  in this repository is the simplified attention implementation. See
  [FlashAttention](../attention/flash_attention.md). `self.att` is kept as a
  backward-compatible alias.
- `self.ffn` — a **list** of two Dense layers: expand to `ff_dim`, project back
  to `embed_dim`.
- `self.layernorm1/2` — pre-attention and pre-FFN norms (or post-norm in the
  post-norm variant).
- `self.dropout1/2` — dropout after attention and after the FFN.

`Layer.sublayers` traverses attributes and lists, so the optimizer finds the
attention, both Dense layers, both norms, and both dropouts automatically.

---

## `build`

```python
def build(self, input_shape):
    self.attention.build(input_shape)
    curr_shape = input_shape
    for layer in self.ffn:
        layer.build(curr_shape)
        curr_shape = (input_shape[0], input_shape[1], layer.units)
    self.layernorm1.build(input_shape)
    self.layernorm2.build(input_shape)
    self.dropout1.build(input_shape)
    self.dropout2.build(input_shape)
    super.build(input_shape)
```

The FFN layers are built in sequence: the first expands the last dimension to
`ff_dim`, the second projects back to `embed_dim`. Norms and dropout do not
change the shape.

Shape walkthrough:

```text
input: (B, S, D)
attention: (B, S, D)
ffn[0]: (B, S, ff_dim)
ffn[1]: (B, S, D)
output: (B, S, D)
```

---

## `forward`

### Causal mask

```python
mask = None
if self.causal:
    mask = BaseAttention.create_causal_mask(inputs.shape[1])
    if kv_cache and layer_id in kv_cache.k_cache:
        q_len = inputs.shape[1]
        kv_len = q_len + kv_cache.k_cache[layer_id].shape[2]
        mask = np.ones((q_len, kv_len))
        mask[:, -q_len:] = BaseAttention.create_causal_mask(q_len)
        mask[:,:-q_len] = 0
```

- During training, the mask is a `(S, S)` upper-triangular matrix. A `1` means
  "not allowed"; attention adds `-1e9` there, so softmax ignores it.
- During cached generation, the mask covers the cached past plus the current
  token. Past tokens are fully visible (`0`); current tokens use the triangular
  mask.

### Pre-norm path

```python
norm1 = self.layernorm1(inputs, training=training)
attn_output = self.attention(norm1, mask=mask, training=training,
                             kv_cache=kv_cache, layer_id=layer_id)
attn_dropped = self.dropout1(attn_output, training=training)
h = inputs + attn_dropped

norm2 = self.layernorm2(h, training=training)
ffn_1 = self.ffn[0](norm2, training=training)
ffn_2 = self.ffn[1](ffn_1, training=training)
ffn_dropped = self.dropout2(ffn_2, training=training)
return h + ffn_dropped
```

The two `+` operations are the residual connections. They give gradients a
direct path from the output back to the input.

### Post-norm path

```python
attn_output = self.attention(inputs, mask=mask, training=training,
                             kv_cache=kv_cache, layer_id=layer_id)
attn_dropped = self.dropout1(attn_output, training=training)
out1 = self.layernorm1(inputs + attn_dropped, training=training)

ffn_1 = self.ffn[0](out1, training=training)
ffn_2 = self.ffn[1](ffn_1, training=training)
ffn_dropped = self.dropout2(ffn_2, training=training)
return self.layernorm2(out1 + ffn_dropped, training=training)
```

---

## Backward

`TransformerBlock` does **not** define `backward`. The base
`Layer.backward` replays `forward` inside a `GradientTape` and computes
gradients for every sublayer. This is the tape-backed path described in
[Layer base class](../base.md#backward-the-tape-backed-fallback).

The residual gradient routing you might expect:

```text
pre-norm:
  d h = d output + d FFN path
  d input = d h + d attention path
post-norm:
  d layernorm2 -> d FFN path and d residual
```

Those sums are computed by the tape's reverse pass, not by hand-written code in
this class.

---

## What to notice

- **The FFN is a list.** `Layer.sublayers` and the optimizer traverse lists,
  so no manual registration is needed.
- **`training` is threaded through every sublayer.** Dropout is only active
  during training.
- **The KV cache is optional.** During training it is `None`; during generation
  it carries past keys/values per layer.
- **Pre-norm vs post-norm changes the data flow.** The backward pass follows
  automatically because it is tape-based.

---

## Read next

- [MultiHeadAttention](../attention/mha.md)
- [FlashAttention (simplified)](../attention/flash_attention.md)
- [LayerNormalization](../normalization/layernorm.md)
- [Dense](../core/dense.md)
- [Layer base class](../base.md)
