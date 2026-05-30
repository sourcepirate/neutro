# RMSNorm and GroupNorm

## RMSNorm

### What does this layer do?

RMSNorm (Root Mean Square Normalization) is a simplified version of LayerNorm that **drops the mean-centering step**. It only divides by the root-mean-square of the activations, then scales by a learned weight.

Why remove the mean? Empirical results from Llama, Qwen, and DeepSeek show that the mean-centering in LayerNorm doesn't help much — the RMS scaling alone provides enough normalization. This saves computation (no mean subtraction) and simplifies the backward pass.

### The math

$$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2 + \epsilon} \quad\quad y = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

No `beta` parameter, no mean computation — just a single learnable `weight` (also called `gamma`).

The RMS is computed `axis=-1` (the feature dimension), same as LayerNorm.

### Walking through the code

#### `__init__` / `build`

```python
def __init__(self, epsilon=1e-6, **kwargs):
    super().__init__(**kwargs)
    self.epsilon = epsilon

def build(self, input_shape):
    self.dim = input_shape[-1]
    self.params['weight'] = np.ones(self.dim)
    super().build(input_shape)
```

🔍 **Only one parameter**: RMSNorm has `weight` but no `beta`. LayerNorm has `gamma` and `beta`. This saves 50% of the normalization parameters.

📐 **`weight`**: Shape `(D,)` — one scalar per feature dimension. Starts as all ones (identity).

#### `forward`

```python
def forward(self, x, training=False):
    self.x = x
    self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.epsilon)
    self.x_norm = x / self.rms
    return self.x_norm * self.params['weight']
```

🔍 **`np.mean(x**2, axis=-1, keepdims=True)`**: Notice we compute `x²` first (element-wise square), then average over the feature axis. This is RMS — no mean-centering.

📐 **`self.rms`**: Shape `(B, S, 1)` for a `(B, S, D)` input. One RMS value per position per sample.

📐 **`x / self.rms`**: Broadcasting divides each feature by its RMS.

📐 **Return**: `(B, S, D)` — same as input, each position normalized independently.

🔍 **`training=False` is accepted but ignored**: RMSNorm works identically at train and inference time. No running statistics, no mode switching. This simplicity is a big advantage over BatchNorm.

#### `backward`

```python
def backward(self, grad_output):
    self.grads['weight'] = np.sum(grad_output * self.x_norm, axis=(0, 1))

    N = self.dim
    grad_x_norm = grad_output * self.params['weight']
    sum_grad_x = np.sum(grad_x_norm * self.x, axis=-1, keepdims=True)
    dx = (grad_x_norm / self.rms) - (self.x * sum_grad_x / (N * self.rms**3))
    return dx
```

📐 **`self.grads['weight']`**: Summing over axes `(0, 1)` — the batch and sequence dimensions. For a `(B, S, D)` gradient, this produces `(D,)`.

🔍 **`grad_x_norm`**: The gradient through the `weight` scale, same concept as `dx_norm` in LayerNorm.

🔍 **The `dx` formula**: Two terms:
1. **`grad_x_norm / rms`**: The direct path — if RMS were just a constant divisor.
2. **`- x * sum(grad_x_norm * x) / (N * rms³)`**: Corrects for the fact that `rms` depends on `x`. When `x` changes, the RMS changes too, and this term accounts for that.

Compare this to the LayerNorm backward and you'll notice it's **simpler** — no term for the mean. The mean subtraction was adding two correction terms (`N * dx_norm` had `-sum(dx_norm)` in LayerNorm); here we just have the direct path and the variance correction.

---

## Group Normalization

### What does this layer do?

Group Normalization (GroupNorm) divides the channels of a convolutional feature map into **groups** and normalizes within each group independently. It's the middle ground between LayerNorm (too coarse for vision) and BatchNorm (too batch-dependent).

Imagine a feature map with 64 channels. With 8 groups, each group has 8 channels. GroupNorm computes mean and variance over `(height, width, 8 channels)` — every spatial position and all channels in the group.

### When to use GroupNorm?

- **Small batch sizes** (video, medical imaging, object detection)
- **Vision transformers** (ViT) where batch size is constrained by memory
- Any time BatchNorm's batch dependency causes trouble

### The math

For input `x` with shape `(N, H, W, C)` and `G` groups, the channels are split: each group has `C // G` channels. For group `g`:

$$\mu_g = \frac{1}{|\mathcal{G}_g|} \sum_{i \in \mathcal{G}_g} x_i \quad\quad \sigma_g^2 = \frac{1}{|\mathcal{G}_g|} \sum_{i \in \mathcal{G}_g} (x_i - \mu_g)^2$$

$$\hat{x} = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} \quad\quad y = \gamma \odot \hat{x} + \beta$$

The mean and variance are computed over axes `(H, W, C//G)` — all spatial positions and all channels within a group.

### Walking through the code

#### `__init__` / `build`

```python
def __init__(self, groups=32, epsilon=1e-5, **kwargs):
    super().__init__(**kwargs)
    self.groups = groups
    self.epsilon = epsilon

def build(self, input_shape):
    dim = input_shape[-1]
    if dim % self.groups != 0:
        raise ValueError(
            f"Number of channels ({dim}) must be divisible by groups ({self.groups})"
        )

    self.params['gamma'] = np.ones((1, 1, 1, dim))
    self.params['beta'] = np.zeros((1, 1, 1, dim))
    super().build(input_shape)
```

🔍 **`groups=32`**: Default from the original paper. For 64 channels, 32 groups means 2 channels per group.

🔍 **`dim % self.groups != 0` check**: Groups must evenly divide channels. If you have 64 channels and 3 groups, that's impossible — you can't split 64 into 3 equal integer parts.

📐 **`gamma` / `beta`**: Shape `(1, 1, 1, C)` — four-dimensional with dummy batch/spatial dims. This makes broadcasting against `(N, H, W, C)` automatic without explicit broadcasting.

#### `forward`

```python
def forward(self, x, training=False):
    self.x_shape = x.shape
    batch, h, w, c = x.shape
    g = self.groups

    x_reshaped = x.reshape(batch, h, w, g, c // g)

    self.mean = np.mean(x_reshaped, axis=(1, 2, 4), keepdims=True)
    self.var = np.var(x_reshaped, axis=(1, 2, 4), keepdims=True)

    self.std = np.sqrt(self.var + self.epsilon)
    self.x_centered = x_reshaped - self.mean
    self.x_norm = self.x_centered / self.std

    x_norm = self.x_norm.reshape(batch, h, w, c)

    return self.params['gamma'] * x_norm + self.params['beta']
```

📐 **`x.reshape(batch, h, w, g, c // g)`**: The key move. An input `(2, 4, 4, 64)` with 8 groups becomes `(2, 4, 4, 8, 8)`. The last dimension is now the group sub-dimension.

🔍 **`np.mean(x_reshaped, axis=(1, 2, 4))`**: Mean over `height (1)`, `width (2)`, and `group-channel (4)`. This computes one mean per **batch element × group** — shape `(2, 1, 1, 8, 1)`.

📐 **`self.mean` shape**: `(N, 1, 1, G, 1)` — one mean per sample per group. Broadcasting divides each group's `H*W*(C//G)` elements by their shared mean.

📐 **`self.var` shape**: Same `(N, 1, 1, G, 1)`.

📐 **Back to `(N, H, W, C)`**: After normalization, reshape back to the original shape.

🔍 **`training=False` is ignored**: Like RMSNorm and LayerNorm, GroupNorm is identical during training and inference. No running statistics.

#### `backward`

```python
def backward(self, grad_output):
    batch, h, w, c = self.x_shape
    g = self.groups
    m = h * w * (c // g)

    self.grads['gamma'] = np.sum(
        grad_output * self.x_norm.reshape(batch, h, w, c),
        axis=(0, 1, 2), keepdims=True
    )
    self.grads['beta'] = np.sum(
        grad_output, axis=(0, 1, 2), keepdims=True
    )

    dx_norm = grad_output * self.params['gamma']
    dx_norm = dx_norm.reshape(batch, h, w, g, c // g)

    sum_dx_norm = np.sum(dx_norm, axis=(1, 2, 4), keepdims=True)
    sum_dx_norm_x_norm = np.sum(dx_norm * self.x_norm, axis=(1, 2, 4), keepdims=True)

    dx = (1.0 / m) / self.std * (
        m * dx_norm - sum_dx_norm - self.x_norm * sum_dx_norm_x_norm
    )

    return dx.reshape(batch, h, w, c)
```

📐 **`m = h * w * (c // g)`**: The total number of elements contributing to each group's statistics. For a `(N, 4, 4, 64)` input with 8 groups: `m = 4 * 4 * 8 = 128`.

📐 **`self.grads['gamma']`**: Sum over `(0, 1, 2)` — batch, height, width. Produces shape `(1, 1, 1, C)` matching the gamma parameter.

📐 **`dx_norm.reshape(batch, h, w, g, c // g)`**: Reshape the gradient into the same grouped format as forward, so we can compute per-group statistics.

🔍 **Sum axes `(1, 2, 4)`**: The same axes used during forward — height, width, and group-channel. `sum_dx_norm` has shape `(N, 1, 1, G, 1)`.

🔍 **The big `dx` formula**: The same structure as BatchNorm and LayerNorm! Three terms:
1. `m * dx_norm` — direct gradient
2. `- sum_dx_norm` — correction through mean
3. `- x_norm * sum_dx_norm_x_norm` — correction through variance

Applied **independently per group** because the sums are over group-specific axes.

## Comparing the four normalizations

| Layer | Stat axes | Parameters | Running stats? | Used in |
|---|---|---|---|---|
| LayerNorm | `(D)` — features | γ, β | No | Transformers |
| BatchNorm | `(N, H, W)` — batch/spatial | γ, β | Yes (mean, var) | CNNs (big batch) |
| RMSNorm | `(D)` — features | weight | No | Llama, DeepSeek |
| GroupNorm | `(H, W, C//G)` — spatial + sub-channels | γ, β | No | Vision (small batch) |

## References

- Zhang, B., & Sennrich, R. (2019). **Root Mean Square Layer Normalization**. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Wu, Y., & He, K. (2018). **Group Normalization**. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
