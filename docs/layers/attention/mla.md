# Multi-Head Latent Attention (MLA)

## Theory

MLA is an attention variant used in DeepSeek models that reduces the KV cache size by compressing keys and values into a latent space. Instead of caching the full $K, V$ projections, MLA caches a compressed latent vector and reconstructs $K, V$ on the fly.

### Standard Attention (per head)

$$Q = XW^Q,\quad K = XW^K,\quad V = XW^V$$

### MLA

$$c_t = \text{RMSNorm}(X_t W^{\text{down}}) \quad \text{(compress to latent)}$$
$$K_t = c_t W^{\text{up}}, \quad V_t = c_t W^{\text{up}} \quad \text{(reconstruct)}$$

The KV cache stores only $c_t$ (latent), not $K_t, V_t$, reducing memory by a factor of $d_{\text{model}} / d_{\text{latent}}$.

## Implementation Guide

### File: `neutro/layers/attention/mla.py`

```python
class MLA(Layer):
    def __init__(self, num_heads, key_dim, latent_dim=128, ...):
```

- `latent_dim`: the compressed representation size (typically much smaller than `num_heads * key_dim`).
- The layer implements both the compression (`W_down`) and reconstruction (`W_up`) projections.
- During forward, it caches the latent `c_t` instead of the full `K, V` tensors.

## Usage Example

```python
from neutro.layers.attention.mla import MLA

mla = MLA(num_heads=8, key_dim=64, latent_dim=32)
x = np.random.randn(2, 16, 512)
y = mla(x)  # shape (2, 16, 512)
```

## References

- DeepSeek-AI. (2024). **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
