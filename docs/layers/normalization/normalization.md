# Normalization Layers

## Theory

Normalization layers stabilize training by controlling the distribution of activations. `neutro` implements four variants.

### Layer Normalization — `neutro/layers/normalization/layernorm.py`

Normalizes across the feature dimension for each sample independently:

$$\mu = \frac{1}{H} \sum_{i=1}^H x_i, \quad \sigma = \sqrt{\frac{1}{H} \sum_{i=1}^H (x_i - \mu)^2 + \epsilon}$$

$$\hat{x} = \frac{x - \mu}{\sigma}, \quad y = \gamma \hat{x} + \beta$$

Used in Transformers (GPT, BERT, Llama). Independent of batch size.

### Batch Normalization — `neutro/layers/normalization/batchnorm.py`

Normalizes across the batch dimension for each feature:

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum x_i, \quad \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum (x_i - \mu_{\mathcal{B}})^2$$

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

Tracks running mean/variance for inference. Used in CNNs.

### RMS Norm — `neutro/layers/normalization/rmsnorm.py`

Root Mean Square Normalization — a simplified LayerNorm without mean centering:

$$\text{RMS}(x) = \sqrt{\frac{1}{H} \sum_{i=1}^H x_i^2 + \epsilon}, \quad y = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

Used in Llama and modern LLMs for efficiency.

### Group Normalization — `neutro/layers/normalization/groupnorm.py`

Divides channels into groups and normalizes within each group:

$$\mu_g = \frac{1}{|\mathcal{G}_g|} \sum_{i \in \mathcal{G}_g} x_i, \quad \sigma_g^2 = \frac{1}{|\mathcal{G}_g|} \sum_{i \in \mathcal{G}_g} (x_i - \mu_g)^2$$

Used in vision models when batch size is small (e.g., video, medical imaging).

## Implementation Guide

All normalization layers share a common pattern:

| Method | Behavior |
|---|---|
| `build(input_shape)` | Allocates `gamma` (scale) and `beta` (shift) parameters. Shape matches the feature dimension. |
| `forward(x)` | Computes mean/variance, normalizes, scales, shifts. |
| `backward(grad)` | Backpropagates through normalization using the stored mean/variance. |

For LayerNorm:

```python
def forward(self, x):
    self.mean = np.mean(x, axis=-1, keepdims=True)
    self.var = np.var(x, axis=-1, keepdims=True)
    self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
    return self.gamma * self.x_hat + self.beta
```

## Usage Example

```python
from neutro.layers import LayerNormalization

ln = LayerNormalization(epsilon=1e-6)
x = np.random.randn(4, 16, 64)  # (batch, seq, features)
y = ln(x)  # Normalized along last axis, same shape
```

## References

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). **Layer Normalization**. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Ioffe, S., & Szegedy, C. (2015). **Batch Normalization**. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Zhang, B., & Sennrich, R. (2019). **Root Mean Square Layer Normalization**. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Wu, Y., & He, K. (2018). **Group Normalization**. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
