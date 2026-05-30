# TransformerBlock

## What does this layer do?

The Transformer block is the fundamental building block of LLMs like GPT and BERT. It combines self-attention with a feed-forward network, using residual connections and layer normalization to keep training stable even in very deep stacks.

## The architecture

Two variants:

### Pre-Norm (modern, used in GPT-2, Llama)

```
output = x + FFN(LayerNorm(x + Attention(LayerNorm(x))))
```

### Post-Norm (original Transformer, used in BERT)

```
output = LayerNorm(x + FFN(LayerNorm(x + Attention(x))))
```

Pre-Norm is more stable during training because the norm is applied BEFORE each sub-layer, keeping activations controlled. Post-Norm was used in the original "Attention Is All You Need" paper but drifted gradients for deep stacks.

## Walking through the code

### Step 1: `__init__` (lines 11–25) — assembling the sub-layers

```python
def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, causal=False, use_flash=False, pre_norm=False, **kwargs):
    super().__init__(**kwargs)
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.causal = causal
    self.pre_norm = pre_norm
    if use_flash:
        self.att = FlashAttention(num_heads=num_heads, key_dim=embed_dim)
    else:
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.ffn = [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)
```

🔍 **Line 17–20**: `self.att = FlashAttention(...)` or `MultiHeadAttention(...)` — the attention sub-layer. The `use_flash` flag lets you pick between regular MHA and the memory-efficient FlashAttention variant.

🔍 **Line 21**: `self.ffn = [Dense(ff_dim, activation='relu'), Dense(embed_dim)]` — a 2-layer MLP stored as a LIST. The first Dense expands from `embed_dim` to `ff_dim` with ReLU, the second projects back to `embed_dim`.

🔍 **Lines 22–23**: `self.layernorm1`, `self.layernorm2` — two layer norms, one before attention and one before the FFN.

🔍 **Lines 24–25**: `self.dropout1`, `self.dropout2` — dropout applied after each sub-layer for regularization.

🧠 "The sublayers are stored as plain attributes (including a list for `ffn`), so `Layer.sublayers` can traverse and find them all automatically for the optimizer."

### Step 2: `build` (lines 27–37) — building the sub-layers in order

```python
def build(self, input_shape):
    self.att.build(input_shape)
    curr_shape = input_shape
    for layer in self.ffn:
        layer.build(curr_shape)
        curr_shape = (input_shape[0], input_shape[1], layer.units)
    self.layernorm1.build(input_shape)
    self.layernorm2.build(input_shape)
    self.dropout1.build(input_shape)
    self.dropout2.build(input_shape)
    super().build(input_shape)
```

🔍 **Lines 29–32**: The `ffn` list is built sequentially. The first `Dense(ff_dim)` takes the embedding shape; the second `Dense(embed_dim)` takes the expanded shape. The `curr_shape` is updated after each build so the next layer knows what to expect.

🔍 **Lines 33–36**: Layer norms and dropouts take the same input shape — they don't change the dimension, they just transform values.

### Step 3: `forward` (lines 42–80) — pre-norm vs post-norm

First, a shared preamble (lines 43–55):

```python
self.inputs = inputs
mask = None
if self.causal:
    mask = BaseAttention.create_causal_mask(inputs.shape[1])
    if kv_cache and layer_id in kv_cache.k_cache:
        q_len = inputs.shape[1]
        kv_len = q_len + kv_cache.k_cache[layer_id].shape[2]
        mask = np.ones((q_len, kv_len))
        mask[:, -q_len:] = BaseAttention.create_causal_mask(q_len)
        mask[:, :-q_len] = 0
```

🔍 **Lines 45–55**: The causal mask prevents attending to future tokens (autoregressive generation). When a KV cache is active, the mask expands to cover all past cached tokens plus the current ones — past tokens are fully visible (0 mask), current tokens use a triangular causal mask.

**Pre-Norm path** (lines 57–68):

```python
if self.pre_norm:
    norm1 = self.layernorm1(inputs, training=training)
    attn_output = self.att(norm1, mask=mask, training=training, kv_cache=kv_cache, layer_id=layer_id)
    attn_dropped = self.dropout1(attn_output, training=training)
    h = inputs + attn_dropped

    norm2 = self.layernorm2(h, training=training)
    ffn_1 = self.ffn[0](norm2, training=training)
    ffn_2 = self.ffn[1](ffn_1, training=training)
    ffn_dropped = self.dropout2(ffn_2, training=training)
    return h + ffn_dropped
```

🔍 **Line 62**: `h = inputs + attn_dropped` — the residual (skip) connection for the attention sub-layer. The original input is added back after attention + dropout, creating a direct gradient highway.

🔍 **Lines 64–67**: The FFN path: first expand with ReLU, then project back. Dropout is applied to the final FFN output.

🔍 **Line 68**: `return h + ffn_dropped` — the second residual connection. The output of the attention path (`h`) is added to the FFN output.

**Post-Norm path** (lines 69–80):

```python
else:
    attn_output = self.att(inputs, mask=mask, training=training, kv_cache=kv_cache, layer_id=layer_id)
    attn_dropped = self.dropout1(attn_output, training=training)
    self.out1_pre_ln = inputs + attn_dropped
    out1 = self.layernorm1(self.out1_pre_ln, training=training)

    ffn_1 = self.ffn[0](out1, training=training)
    ffn_2 = self.ffn[1](ffn_1, training=training)
    ffn_dropped = self.dropout2(ffn_2, training=training)
    self.out2_pre_ln = out1 + ffn_dropped
    return self.layernorm2(self.out2_pre_ln, training=training)
```

🔍 **Lines 71–73**: Attention is applied FIRST, then dropout, then the residual is added. Only AFTER the residual is the layer norm applied. This is the reverse order from pre-norm.

🔍 **Lines 79–80**: Same pattern for the FFN — add residual first, then normalize at the very end.

🔍 **Lines 73 & 79**: `self.out1_pre_ln` and `self.out2_pre_ln` are stored as attributes. These are intermediates needed by the backward pass (since post-norm applies norm after the residual, backward needs the pre-norm values).

### Step 4: `backward` (lines 82–107) — manual skip-connection gradient routing

**Pre-Norm backward** (lines 83–96):

```python
if self.pre_norm:
    grad_ffn_path = self.dropout2.backward(grad_output)
    grad_ffn = self.ffn[1].backward(grad_ffn_path)
    grad_ffn = self.ffn[0].backward(grad_ffn)
    grad_norm2 = self.layernorm2.backward(grad_ffn)
    grad_h = grad_output + grad_norm2

    grad_attn_path = self.dropout1.backward(grad_h)
    grad_attn = self.att.backward(grad_attn_path)
    grad_norm1 = self.layernorm1.backward(grad_attn)
    return grad_h + grad_norm1
```

🔍 **Line 90**: `grad_h = grad_output + grad_norm2` — the skip connection in backward. `grad_output` is the gradient flowing directly through the skip connection (bypassing the FFN). `grad_norm2` is the gradient that came through the FFN path. They sum at the branch point. This is what makes training deep networks possible!

🔍 **Line 96**: `return grad_h + grad_norm1` — same pattern again. The gradient splits at the first residual connection: one copy goes straight back to the input (via the skip), the other goes through the attention sub-layer.

🔍 "Without skip connections, gradients would shrink through every layer (vanishing gradient). With skips, the gradient has a direct path from output to input — notice how `grad_output` appears directly in the return value on line 96."

**Post-Norm backward** (lines 97–107):

```python
else:
    grad = self.layernorm2.backward(grad_output)
    grad_ffn_path = self.dropout2.backward(grad)
    grad_ffn = self.ffn[1].backward(grad_ffn_path)
    grad_ffn = self.ffn[0].backward(grad_ffn)
    grad_out1 = self.layernorm1.backward(grad + grad_ffn)

    grad_attn_path = self.dropout1.backward(grad_out1)
    grad_attn = self.att.backward(grad_attn_path)
    return grad_out1 + grad_attn
```

🔍 **Line 99**: The first thing backward does is go through `layernorm2` — the reverse of the forward order where layernorm was the last step.

🔍 **Line 103**: `grad + grad_ffn` combines the gradient from the second residual connection before passing through `layernorm1`'s backward.

🔍 **Line 107**: `return grad_out1 + grad_attn` — the gradient from the first residual connection. Notice how the post-norm backward stores intermediates (`self.out1_pre_ln`) that were computed during forward — this is needed because `LayerNormalization.backward` uses the pre-norm input to compute its gradient.

## Putting it all together

- The `TransformerBlock` contains 7 sublayers: 1 attention layer, 2 Dense layers (in a list), 2 layer norms, and 2 dropout layers.
- These sublayers are automatically discovered by `Layer.sublayers` because they're stored as attributes.
- They're built individually in `build` since each expects a different input shape (especially the two Dense layers in `ffn`).
- Parameters are collected by the optimizer via `_get_all_layers`, which traverses the sublayer tree.
- The block handles routing between sublayers manually (forward calls each one in sequence; backward reverses the sequence) rather than delegating to a `Model`.

## Common patterns

- **"The `ffn` is a `list` of `Dense` layers"** — this is why `Layer.sublayers` has special handling to iterate over lists found as attributes.
- **"Dropout is only active during training"** — the `training` flag is threaded through every sub-layer call. At inference, dropout is a no-op.
- **"The KV cache is optional"** — it's only populated and used during autoregressive generation. During training, `kv_cache` is `None`.
- **"Pre-norm vs post-norm changes the forward AND backward pass"** — both the data flow and gradient flow are inverted. Post-norm also needs intermediate buffers (`out1_pre_ln`, `out2_pre_ln`) for the backward pass.
