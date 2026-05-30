# Batch Normalization

## What does this layer do?

Batch Normalization stabilizes training by normalizing activations **across the batch dimension** instead of across features. For each feature channel, it computes the mean and variance over the entire batch, then normalizes and applies a learned scale (`gamma`) and shift (`beta`).

Intuitively, BatchNorm says: "Every feature channel should have a consistent distribution across batch elements." This is great for CNNs and large batch sizes, but it breaks when the batch is small (e.g., medical imaging, video).

During **inference**, BatchNorm uses running statistics collected during training — a single sample can't give a meaningful batch mean.

## The math

For a mini-batch with shape `(batch, ..., D)`, mean and variance are computed over **all axes except the last**:

$$\mu = \frac{1}{M} \sum_{i=1}^{M} x_i \quad\quad \sigma^2 = \frac{1}{M} \sum_{i=1}^{M} (x_i - \mu)^2$$

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \quad\quad y = \gamma \odot \hat{x} + \beta$$

The key difference from LayerNorm: the sum runs over all batch/spatial elements (all axes except the feature axis `axis=-1`), producing stats of shape `(D,)` — one mean and variance per feature channel.

## Walking through the code

### `__init__` / `build`

```python
def __init__(self, momentum=0.99, epsilon=1e-3):
    super().__init__()
    self.momentum = momentum
    self.epsilon = epsilon
    self.running_mean = None
    self.running_var = None

def build(self, input_shape):
    dim = input_shape[-1]
    self.params['gamma'] = np.ones(dim)
    self.params['beta'] = np.zeros(dim)
    self.running_mean = np.zeros(dim)
    self.running_var = np.ones(dim)
    super().build(input_shape)
```

🔍 **`momentum=0.99`**: Controls how fast running statistics update. With momentum 0.99, each new batch contributes 1% to the running average. Higher = more stable but slower to adapt.

🔍 **`epsilon=1e-3`**: Notice this is larger than LayerNorm's `1e-6`. BatchNorm variance tends to be noisier (fewer elements in each statistic), so a slightly larger epsilon is common.

🔍 **`running_mean` / `running_var`**: Start as `zeros` and `ones` — a neutral initialization. These accumulate statistics across all training batches.

📐 **`gamma`, `beta`**: Shape `(D,)` — one per feature channel. Same as LayerNorm.

### `forward`

```python
def forward(self, x, training=False):
    if training:
        mean = np.mean(x, axis=tuple(range(len(x.shape)-1)))
        var = np.var(x, axis=tuple(range(len(x.shape)-1)))

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        self.x_centered = x - mean
        self.std = np.sqrt(var + self.epsilon)
        self.x_norm = self.x_centered / self.std
    else:
        x_centered = x - self.running_mean
        std = np.sqrt(self.running_var + self.epsilon)
        self.x_norm = x_centered / std

    return self.params['gamma'] * self.x_norm + self.params['beta']
```

🔍 **Training / Inference split**: The layer behaves completely differently depending on `training`.

**Training path:**

📐 **`np.mean(x, axis=tuple(range(len(x.shape)-1)))`**: For a `(B, H, W, D)` input, this sums over axes `(0, 1, 2)` — all but the last. Result shape: `(D,)` — one mean per feature channel.

📐 **Same for `var`**: Shape `(D,)`.

🔍 **Running stats update**: The exponential moving average:
- `running_mean = 0.99 * old + 0.01 * batch_mean`
- Over time, this smooths out the noise of individual batches.

🔍 **Why running stats?** At inference, you might get a single image `(1, H, W, D)`. That batch mean of 1 element is meaningless. So you use the running statistics that accumulated statistics over thousands of training samples.

**Inference path:**

No caching of `x_centered`, `std`, or `x_norm`. Inference just computes the output directly using `running_mean` and `running_var`.

### `backward`

```python
def backward(self, grad_output):
    gamma = self.params['gamma']
    batch_size = np.prod(grad_output.shape[:-1])

    self.grads['gamma'] = np.sum(grad_output * self.x_norm,
        axis=tuple(range(len(grad_output.shape)-1)))
    self.grads['beta'] = np.sum(grad_output,
        axis=tuple(range(len(grad_output.shape)-1)))

    dx_norm = grad_output * gamma
    dx = (1. / batch_size) / self.std * (
        batch_size * dx_norm
        - np.sum(dx_norm, axis=tuple(range(len(grad_output.shape)-1)))
        - self.x_norm * np.sum(dx_norm * self.x_norm,
            axis=tuple(range(len(grad_output.shape)-1)))
    )
    return dx
```

📐 **`batch_size = np.prod(grad_output.shape[:-1])`**: The total number of elements contributing to each feature's statistics. For `(B, H, W, D)`, this is `B * H * W` — all batch and spatial positions.

📐 **`self.grads['gamma']`**: Shape `(D,)` — sum over axes `(0, 1, 2)` for a 4D input. Same pattern as LayerNorm but the reduction axes are different.

📐 **`self.grads['beta']`**: Same reduction, also `(D,)`.

🔍 **Big `dx` formula**: Identical structure to LayerNorm! The three terms are the same:
1. `batch_size * dx_norm` — direct gradient
2. `- sum(dx_norm)` — correction through mean
3. `- x_norm * sum(dx_norm * x_norm)` — correction through variance

The only difference is **which axis** the sums are computed over. For BatchNorm, sums are over all non-feature axes. For LayerNorm, sums are only over the feature axis.

🔍 **`self.std` shape**: `(D,)` for BatchNorm vs. `(B, S, 1)` for LayerNorm. But broadcasting makes the math work the same way.

## Why not always use BatchNorm?

BatchNorm has two gotchas:
1. **Small batches**: Mean/variance from 2-4 samples is noisy.
2. **Training != Inference**: You must track running stats. A bug in mode-switching silently destroys accuracy.

That's why LayerNorm is preferred in Transformers and GroupNorm is used in vision with small batches.

## References

- Ioffe, S., & Szegedy, C. (2015). **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**. *Proceedings of ICML*. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
