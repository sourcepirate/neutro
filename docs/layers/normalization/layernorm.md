# Layer Normalization

## What does this layer do?

Layer Normalization (LayerNorm) makes training stable by controlling the distribution of activations inside a neural network. For each input sample, it rescales all features to have zero mean and unit variance, then applies a learned scale (`gamma`) and shift (`beta`).

Unlike Batch Normalization, LayerNorm computes statistics **across the feature dimension** — independently for every sample in the batch. This makes it batch-size agnostic, which is why every Transformer (BERT, GPT, Llama) uses it.

## The math

For an input `x` with shape `(..., D)` (the last dimension is the feature dimension):

$$\mu = \frac{1}{D} \sum_{i=1}^{D} x_i \quad\quad \sigma^2 = \frac{1}{D} \sum_{i=1}^{D} (x_i - \mu)^2$$

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \quad\quad y = \gamma \odot \hat{x} + \beta$$

The mean and variance are computed over the **last axis** (`axis=-1`). Every sample in a batch gets its own normalization. The `gamma` and `beta` vectors have the same size as the feature dimension.

## Walking through the code

### `__init__` / `build`

```python
def __init__(self, epsilon=1e-6, **kwargs):
    super().__init__(**kwargs)
    self.epsilon = epsilon

def build(self, input_shape):
    self.params['gamma'] = np.ones(input_shape[-1])
    self.params['beta'] = np.zeros(input_shape[-1])
    super().build(input_shape)
```

🔍 **`__init__`**: Stores a tiny `epsilon` to prevent division by zero during normalization. The `**kwargs` passes layer name and other metadata to `super()`.

🔍 **`build`**: Called once when the layer first sees data. Creates two trainable parameters:
- `gamma = ones(D)` — starts as the identity scaling (multiply by 1)
- `beta = zeros(D)` — starts as the identity shift (add 0)

📐 **Shapes**: `(D,)` — one scale and one shift per feature. If input is `(batch, seq_len, 512)`, then `gamma` and `beta` are both `(512,)`. NumPy broadcasting expands them to match the full input shape during `forward`.

### `forward`

```python
def forward(self, x, training=False):
    self.x = x
    self.mean = np.mean(x, axis=-1, keepdims=True)
    self.var = np.var(x, axis=-1, keepdims=True)
    self.x_norm = (x - self.mean) / np.sqrt(self.var + self.epsilon)
    return self.params['gamma'] * self.x_norm + self.params['beta']
```

🔍 **Line `self.x = x`**: Saves the input for the backward pass. Without this, backward wouldn't know what `x` was during forward.

📐 **`np.mean(x, axis=-1, keepdims=True)`**: Computes mean over the last (feature) axis. With `keepdims`, a `(B, S, D)` input produces a `(B, S, 1)` mean — broadcasting back to `(B, S, D)` when you subtract it.

📐 **Same for `np.var`**: Shape `(B, S, 1)` — one variance per position per sample.

📐 **`self.x_norm`**: Shape `(B, S, D)` — same as input. Each feature `x[b,s,d]` has been centered and scaled by its own sample/position stats.

📐 **Return**: `gamma * x_norm + beta` → still `(B, S, D)`. Broadcasting multiplies `(D,)` gamma against every position.

🔍 **Why cache `mean`, `var`, `x_norm`?** Backward needs them. The gradient through normalization depends on the original mean and variance — you can't recompute them because they're functions of the original `x`.

### `backward`

```python
def backward(self, grad_output):
    N = grad_output.shape[-1]
    self.grads['gamma'] = np.sum(grad_output * self.x_norm,
        axis=tuple(range(len(grad_output.shape)-1)))
    self.grads['beta'] = np.sum(grad_output,
        axis=tuple(range(len(grad_output.shape)-1)))
    dx_norm = grad_output * self.params['gamma']
    std_inv = 1.0 / np.sqrt(self.var + self.epsilon)
    dx = (1.0 / N) * std_inv * (
        N * dx_norm
        - np.sum(dx_norm, axis=-1, keepdims=True)
        - self.x_norm * np.sum(dx_norm * self.x_norm, axis=-1, keepdims=True)
    )
    return dx
```

🔍 **`N = grad_output.shape[-1]`**: The feature dimension. Used in the normalization factor later.

📐 **`self.grads['gamma']`**: Shape `(D,)` — sum over all axes except the last. For a `(B, S, D)` gradient, we sum over axes 0 and 1 (batch and sequence).

📐 **`self.grads['beta']`**: Shape `(D,)` — the gradient of the bias term is just the sum of the output gradient along all non-feature axes.

🔍 **`dx_norm = grad_output * gamma`**: The gradient flowing through the `gamma` scale. If the layer output is `y = gamma * x_norm + beta`, then `dy/dx_norm = gamma`. This is the chain rule's first step: `dL/dx_norm = dL/dy * dy/dx_norm`.

🔍 **The big `dx` formula**: This is the gradient through the entire normalization pipeline — through the standardization `(x - mu) / std`. The formula has three terms inside the parentheses:

1. **`N * dx_norm`**: The direct path — what you'd get if normalization were just a scaling.
2. **`- sum(dx_norm)`**: Corrects for the mean subtraction — the gradient has to account for the fact that `mu` is a function of `x`.
3. **`- x_norm * sum(dx_norm * x_norm)`**: Corrects for the variance scaling — `std` is also a function of `x`.

Think of it as:
```
x → [center: x - μ] → [scale: /σ] → x̂ → [affine: γ·x̂ + β] → y
                    ↑                  ↑                     ↑
               μ = mean(x)        σ = sqrt(var(x))       γ, β learned
```

The backward pass reverses this chain. You don't need to memorize the big formula — just know that it's the chain rule correctly threaded through the mean and variance dependencies.

## References

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). **Layer Normalization**. *arXiv preprint arXiv:1607.06450*. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
