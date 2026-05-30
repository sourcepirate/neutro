# Transformer Block

## Theory

The Transformer block is the fundamental building block of modern LLMs. It combines multi-head attention with a feed-forward network, residual connections, and layer normalization.

### Pre-Norm Architecture

$$\text{output} = x + \text{FFN}(\text{LN}(x + \text{Attention}(\text{LN}(x))))$$

Each sub-layer has a residual connection (`x + sublayer(x)`), which helps gradient flow during backpropagation.

### Post-Norm Architecture (original Transformer)

$$\text{output} = \text{LN}(x + \text{FFN}(\text{LN}(x + \text{Attention}(x))))$$

## Implementation Guide

### File: `neutro/layers/transformer/transformer_block.py`

### `__init__` — line 11

```python
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,
                 causal=False, use_flash=False, pre_norm=False, **kwargs):
```

- `embed_dim`: model dimension (e.g., 768 for GPT-2 small).
- `num_heads`: number of attention heads (must divide `embed_dim`).
- `ff_dim`: feed-forward hidden dimension (typically 4× `embed_dim`).
- `causal`: if True, creates a causal attention mask (for autoregressive generation).
- `use_flash`: if True, uses `FlashAttention` instead of standard `MultiHeadAttention`.
- `pre_norm`: if True, uses Pre-Norm (modern); if False, uses Post-Norm (original).

### Forward pass — line 42

For Pre-Norm:

```python
norm1 = self.layernorm1(inputs, training)
attn_output = self.att(norm1, mask=mask, training=training)
h = inputs + self.dropout1(attn_output, training=training)

norm2 = self.layernorm2(h, training=training)
ffn_output = self.ffn[1](self.ffn[0](norm2, training=training), training=training)
return h + self.dropout2(ffn_output, training=training)
```

The block contains 7 sublayers: `att`, `layernorm1`, `layernorm2`, `dropout1`, `dropout2`, and two Dense layers in `ffn`.

### Backward pass — line 82

The backward manually routes gradients through the skip connections:

```python
def backward(self, grad_output):
    grad_ffn_path = self.dropout2.backward(grad_output)
    grad_ffn = self.ffn[1].backward(grad_ffn_path)
    grad_ffn = self.ffn[0].backward(grad_ffn)
    grad_norm2 = self.layernorm2.backward(grad_ffn)
    grad_h = grad_output + grad_norm2  # Skip connection

    grad_attn_path = self.dropout1.backward(grad_h)
    grad_attn = self.att.backward(grad_attn_path)
    grad_norm1 = self.layernorm1.backward(grad_attn)
    return grad_h + grad_norm1          # Skip connection
```

### Sub-layers

The block exposes its sublayers via the `sublayers` property, which is critical for:
- **Optimizer**: `_get_all_layers` finds them for parameter updates.
- **Shared layer state**: `_capture_layer_state` saves their internal state (inputs, z, etc.) per node.

## Usage Example

```python
from neutro.layers.transformer import TransformerBlock

block = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048, pre_norm=True)
x = np.random.randn(2, 16, 512)  # (batch, seq, embed)
y = block(x)                      # Same shape
```

## References

- Vaswani, A., et al. (2017). **Attention Is All You Need**. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Pre-Norm: GPT-2 / Llama architecture variant.
