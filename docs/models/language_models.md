# Language Models

## GPT — `neutro/models/language/gpt.py`

### GPT-1 (line 5)

$$P(w_t | w_{<t}) = \text{Softmax}(W \cdot h_t)$$

- **Embedding**: Learned token embeddings (`Embedding(vocab_size, dim)`).
- **Position**: Uses learned positional embeddings (not shown, simplified in `neutro`).
- **Blocks**: `TransformerBlock` with Post-Norm (original GPT-1 style), causal masking.
- **Output**: Dense projection to vocabulary, Softmax.

### GPT-2 (line 24)

Same structure but moves to **Pre-Norm** (LayerNorm before each sub-layer, not after), which improves training stability:

```python
# GPT-1: Post-Norm
model.add(TransformerBlock(dim, n_heads, causal=True, pre_norm=False))

# GPT-2: Pre-Norm
model.add(TransformerBlock(dim, n_heads, causal=True, pre_norm=True))
```

GPT-2 also adds a final LayerNorm before the output projection.

## Llama — `neutro/models/language/llama.py`

### Key Innovations

1. **RMSNorm** instead of LayerNorm (faster, comparable quality).
2. **SwiGLU** activation in the MLP: $\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot (xW_3)) W_2$.
3. **RoPE** (Rotary Position Embedding) instead of learned positional embeddings.
4. **Pre-Norm** architecture.

### `LlamaBlock` — line 59

```python
class LlamaBlock(Layer):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, hidden_dim):
        self.attention = FlashAttention(num_heads=n_heads, key_dim=dim, use_rope=True)
        self.attention_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        self.feed_forward = LlamaMLP(dim, hidden_dim)
```

### `LlamaMLP` — line 12

The SwiGLU MLP: three weight matrices $W_1, W_2, W_3$ instead of the standard two:

```python
def forward(self, x):
    gate = self.w1(x)           # Project up
    gate = self.silu(gate)      # Activate with SiLU
    value = self.w3(x)          # Another projection
    multiplied = gate * value   # Element-wise gating
    return self.w2(multiplied)  # Project back down
```

### `LlamaTiny` — line 103

A minimal Llama for educational purposes:

```python
model = Sequential([Embedding(vocab_size, dim)])
for _ in range(n_layers):
    model.add(LlamaBlock(dim, n_heads, n_heads, dim // n_heads, dim * 4))
model.add(RMSNorm())
model.add(Dense(vocab_size))
```

## DeepSeek — `neutro/models/language/deepseek.py`

### Architecture variants

| Function | Attention | Feed-Forward | Key Feature |
|---|---|---|---|
| `DeepSeekV1Tiny` | MHA (FlashAttention) | MoE (8 experts, top-2) | Strong baseline |
| `DeepSeekV2Tiny` | **MLA** (Multi-Head Latent Attention) | MoE (8 experts, top-2) | KV cache reduction |
| `DeepSeekV3Tiny` | MLA | MoE (16 experts, top-4) | More experts, refined routing |

### `DeepSeekMoEBlock` — line 11

Each block contains:
- **Attention**: Either standard `FlashAttention` or `MultiHeadLatentAttention` (MLA).
- **MoE**: `MoELayer` with `num_experts` experts, `top_k` routing.
- **Shared Expert**: A single Dense layer that is always active (complementary to sparse MoE).
- **Residual connections** around both attention and FFN.

```python
def forward(self, x):
    # Attention path
    h = x + self.attention(self.attention_norm(x))
    # MoE + Shared expert
    out = h + self.moe(self.ffn_norm(h)) + self.shared_expert(self.ffn_norm(h))
    return out
```

## Qwen — `neutro/models/language/qwen.py`

Qwen models follow a similar architecture to Llama (RMSNorm, RoPE, SwiGLU) with specific configuration defaults for the Qwen series.

## Usage Example

```python
from neutro.models.language import LlamaTiny, GPT2, DeepSeekTiny
import numpy as np

llama = LlamaTiny(vocab_size=32000, seq_len=128, dim=512, n_layers=4, n_heads=8)
tokens = np.array([[1, 5, 23, 42]])
logits = llama(tokens)  # (1, 4, 32000)

gpt = GPT2(vocab_size=50000, seq_len=64)
logits = gpt(tokens)

deepseek = DeepSeekTiny(vocab_size=32000, seq_len=128)
logits = deepseek(tokens)
```

## References

- Radford, A., et al. (2018). **Improving Language Understanding by Generative Pre-Training** (GPT-1).
- Radford, A., et al. (2019). **Language Models are Unsupervised Multitask Learners** (GPT-2).
- Touvron, H., et al. (2023). **Llama 2: Open Foundation and Fine-Tuned Chat Models**. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- DeepSeek-AI. (2024). **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- Bai, J., et al. (2023). **Qwen Technical Report**. [arXiv:2309.16609](https://arxiv.org/abs/2309.16609)
