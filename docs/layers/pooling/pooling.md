# Pooling Layers

## MaxPooling2D

### What does this layer do?

Slides a fixed-size window over a 2D feature map and keeps only the **maximum** value in each window. This downsamples the spatial dimensions (height & width) while preserving the important features — if a strong edge detector fires somewhere, it doesn't matter exactly *where*.

### The math

For each window at position `(i, j)` in channel `k`:

$$y_{i,j,k} = \max_{p=0..P-1,\; q=0..Q-1} x_{i \cdot s + p,\; j \cdot s + q,\; k}$$

Where $P\times Q$ is the pool window size and $s$ is the stride.

### Walking through the code

#### `__init__` — setting up window geometry

```python
def __init__(self, pool_size=(2, 2), strides=None, data_format='channels_last', **kwargs):
    super().__init__(**kwargs)
    self.pool_size = pool_size if isinstance(pool_size, (tuple, list)) else (pool_size, pool_size)
    strides = strides if strides else self.pool_size
    self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    if data_format not in ('channels_last', 'channels_first'):
        raise ValueError("data_format must be 'channels_last' or 'channels_first'")
    self.data_format = data_format
```

🔍 **Line 3-4**: If `pool_size` is an int like `2`, expand it to `(2, 2)`. Same for `strides`.

🔍 **Line 5**: If `strides` is `None`, default to `pool_size` — the usual "non-overlapping windows" behavior.

🔍 **Lines 7-8**: `data_format` tells us whether channels are last (NHWC — TensorFlow convention) or first (NCHW — PyTorch convention). The layer normalizes everything to `channels_last` internally for simpler indexing: `(batch, h, w, c)`.

#### `forward` — finding the max

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    inputs_nhwc = self._to_channels_last(inputs)
    batch, h, w, c = inputs_nhwc.shape
    ph, pw = self.pool_size
    sh, sw = self.strides

    oh = (h - ph) // sh + 1
    ow = (w - pw) // sw + 1

    x = inputs_nhwc.transpose(0, 3, 1, 2).reshape(-1, 1, h, w)

    self.x_cols = im2col_indices(x, ph, pw, padding=0, stride=sh)
    self.arg_max = np.argmax(self.x_cols, axis=0)
    out = self.x_cols[self.arg_max, np.arange(self.arg_max.size)]

    out = out.reshape(oh, ow, batch, c).transpose(2, 0, 1, 3)
    return self._from_channels_last(out)
```

🔍 **Line 3**: Cache the original input for backward pass. We'll need it for shape information.

🔍 **Line 4**: Normalize to `(batch, h, w, c)` if in `channels_first` format.

📐 **Shape so far**: `inputs_nhwc` is `(B, H, W, C)`.

🔍 **Lines 8-9**: Compute output spatial dimensions. For `H=28, ph=2, sh=2`: `(28-2)//2 + 1 = 14`.

🔍 **Line 10**: Transpose to `(B, C, H, W)` then reshape to `(B*C, 1, H, W)`. This merges batch and channel dimensions so `im2col_indices` treats each channel of each sample independently.

📐 **Shape**: `(B, H, W, C)` → `.transpose(0, 3, 1, 2)` → `(B, C, H, W)` → `.reshape(-1, 1, H, W)` → `(B*C, 1, H, W)`.

🔍 **Line 12**: `im2col_indices` "unrolls" each window into a column. For a 2×2 window, each column has 4 elements (one per window position). The result has shape:

📐 **Shape of `self.x_cols`**: `(ph * pw * in_channels, oh * ow * batch)` = `(4, oh * ow * B)` for a 2×2 pool with 1 channel per grouped sample. Each column is one window; each row is one position *within* that window.

🔍 **Line 13**: `np.argmax(self.x_cols, axis=0)` — for each column (window), find which row (position within the window) has the maximum value.

📐 **Shape of `self.arg_max`**: `(oh * ow * B*C,)` — one index per window. This is cached for backward: it tells us *exactly which element* in each window was the winner.

🔍 **Line 14**: Select the max value from each column using fancy indexing. `self.arg_max` gives the row index, `np.arange(self.arg_max.size)` gives the column index.

📐 **Shape**: `self.x_cols[self.arg_max, np.arange(self.arg_max.size)]` → `(oh * ow * B * C,)` — one scalar per window.

🔍 **Lines 16-17**: Unflatten back: `(oh * ow * B * C,)` → `(oh, ow, B, C)` → `.transpose(2, 0, 1, 3)` → `(B, oh, ow, C)`. Then convert back to original data format if needed.

#### `backward` — routing gradients to the winner

```python
def backward(self, grad_output):
    grad_output_nhwc = self._to_channels_last(grad_output)
    batch, oh, ow, c = grad_output_nhwc.shape
    ph, pw = self.pool_size
    sh, sw = self.strides
    _, h, w, _ = self._shape_to_channels_last(self.inputs.shape)

    dout = grad_output_nhwc.transpose(1, 2, 0, 3).flatten()

    dx_cols = np.zeros_like(self.x_cols)
    dx_cols[self.arg_max, np.arange(self.arg_max.size)] = dout

    dx = col2im_indices(dx_cols, (batch * c, 1, h, w), ph, pw, padding=0, stride=sh)
    return self._from_channels_last(dx.reshape(batch, c, h, w).transpose(0, 2, 3, 1))
```

🔍 **Line 2-3**: Normalize gradient to NHWC and pull out shapes.

🔍 **Line 6**: Get the original input's spatial dimensions `(H, W)` from the cached `self.inputs`.

🔍 **Line 8**: Flatten the gradient to match the column layout from forward.

📐 **Shape**: `grad_output` `(B, oh, ow, C)` → `.transpose(1, 2, 0, 3)` → `(oh, ow, B, C)` → `.flatten()` → `(oh * ow * B * C,)`.

🔍 **Lines 10-11**: The key insight: **only the max position gets the gradient**. Create a zero gradient buffer the same shape as `self.x_cols` (`(4, oh * ow * B * C)` for 2×2 pool). Then, for each window (column), place the output gradient at exactly the row index that was the argmax in forward. All other positions within that window get 0.

🔍 **Why this works**: If element `(0, 0)` was the max in a 2×2 window, `self.arg_max` for that window is `0`. The gradient for position `(0, 0)` equals the output gradient; positions `(0, 1)`, `(1, 0)`, `(1, 1)` in that window get 0. Small changes to non-max positions don't affect the output, so they get no gradient.

📐 **`dx_cols[argmax, col_indices] = dout`**: `dx_cols` has shape `(win_size, n_windows)`. `self.arg_max` has shape `(n_windows,)`. For each window `j`, we set `dx_cols[arg_max[j], j] = dout[j]`.

🔍 **Line 13**: `col2im_indices` is the inverse of `im2col_indices` — it scatters the columns back onto the original `(B*C, 1, H, W)` grid. `np.add.at` is used internally to handle overlapping windows (though with stride == pool_size, there are no overlaps).

🔍 **Line 14**: Reshape from `(B*C, 1, H, W)` back to `(B, C, H, W)` then transpose to `(B, H, W, C)`, and finally convert to original data format.

---

## GlobalAveragePooling2D

### What does this layer do?

Takes each feature map (H×W) and replaces it with a single number — the **average** of all values in that map. This is a dramatic downsampling: `(B, H, W, C)` → `(B, C)`. It's commonly used before the final classification layer in CNNs to replace Flatten + Dense, drastically reducing parameters.

### The math

$$y_k = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j,k}$$

Spatial dimensions are averaged away; only the channel dimension survives.

### Walking through the code

#### `forward`

```python
def forward(self, inputs, training=False):
    self.input_shape_internal = inputs.shape
    if self.data_format == 'channels_first':
        return np.mean(inputs, axis=(2, 3))
    return np.mean(inputs, axis=(1, 2))
```

🔍 **Line 2**: Cache the full input shape. We'll need `H` and `W` in backward to divide the gradient.

🔍 **Lines 3-4**: `np.mean(inputs, axis=(2, 3))` — average over height and width (axes 2 and 3) for `channels_first`.

📐 **Shape**: `(B, C, H, W)` → `np.mean(axis=(2,3))` → `(B, C)`. Each feature map becomes one number.

🔍 **Line 5**: For `channels_last`, spatial dimensions are axes 1 and 2.

📐 **Shape**: `(B, H, W, C)` → `np.mean(axis=(1,2))` → `(B, C)`.

#### `backward`

```python
def backward(self, grad_output):
    if self.data_format == 'channels_first':
        batch, c, h, w = self.input_shape_internal
        return (grad_output[:, :, None, None] * np.ones((batch, c, h, w))) / (h * w)
    batch, h, w, c = self.input_shape_internal
    return (grad_output[:, None, None, :] * np.ones((batch, h, w, c))) / (h * w)
```

🔍 **Line 3 or 6**: Unpack the cached input shape to get `H` and `W`.

🔍 **Lines 4 or 7**: The gradient of an average is the upstream gradient **divided evenly** across all H×W positions. The upstream gradient `grad_output` has shape `(B, C)`. We need to broadcast it back to `(B, H, W, C)` — each of the H×W spatial positions gets the same gradient value, scaled by `1/(H*W)`.

📐 **Shape walkthrough** (channels_last case): `grad_output` `(B, C)` → `grad_output[:, None, None, :]` → `(B, 1, 1, C)` → `* np.ones((B, H, W, C))` → `(B, H, W, C)` → `/(H*W)` → `(B, H, W, C)` — matching the original input shape.

🔍 **Why it works**: If a feature map was 4×4, the forward averaged all 16 values. So each of those 16 values contributed equally. The backward spreads the gradient back equally: each pixel gets 1/16 of the output gradient.

---

## GlobalMaxPooling2D

### What does this layer do?

Same idea as GlobalAveragePooling2D, but instead of averaging, it takes the **maximum** value from each feature map. `(B, H, W, C)` → `(B, C)`.

### The math

$$y_k = \max_{i, j} x_{i,j,k}$$

### Walking through the code

#### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    if self.data_format == 'channels_first':
        inputs_nhwc = inputs.transpose(0, 2, 3, 1)
    else:
        inputs_nhwc = inputs
    self.max_indices = np.argmax(inputs_nhwc.reshape(inputs_nhwc.shape[0], -1, inputs_nhwc.shape[-1]), axis=1)
    return np.max(inputs_nhwc, axis=(1, 2))
```

🔍 **Line 2**: Cache input for backward.

🔍 **Lines 3-5**: Normalize to NHWC for consistent indexing.

🔍 **Line 6**: Flatten spatial dimensions and find the argmax per channel.

📐 **Shape**: `inputs_nhwc` is `(B, H, W, C)`. `.reshape(B, -1, C)` → `(B, H*W, C)`. `np.argmax(axis=1)` → `(B, C)`. Each element `max_indices[b, c]` is the flat spatial index (0 to H*W-1) of the maximum value in channel `c`.

🔍 **Line 7**: `np.max(inputs_nhwc, axis=(1, 2))` — take the max over both spatial dimensions.

📐 **Shape**: `(B, H, W, C)` → `np.max(axis=(1,2))` → `(B, C)`.

#### `backward`

```python
def backward(self, grad_output):
    if self.data_format == 'channels_first':
        batch, c, h, w = self.inputs.shape
        dx_nhwc = np.zeros((batch, h, w, c), dtype=self.inputs.dtype)
    else:
        batch, h, w, c = self.inputs.shape
        dx_nhwc = np.zeros_like(self.inputs)
    for b in range(batch):
        for channel in range(c):
            idx = self.max_indices[b, channel]
            ih, iw = divmod(idx, w)
            dx_nhwc[b, ih, iw, channel] = grad_output[b, channel]
    if self.data_format == 'channels_first':
        return dx_nhwc.transpose(0, 3, 1, 2)
    return dx_nhwc
```

🔍 **Lines 3-6**: Create a zero gradient buffer the same shape as the input. The gradient starts as all zeros — only the max positions will get filled in.

🔍 **Lines 7-10**: For each sample and each channel, find which spatial position `(ih, iw)` was the maximum. `divmod(idx, w)` converts the flat index back to 2D coordinates. Then place `grad_output[b, channel]` at exactly that position.

🔍 **Why no gradient elsewhere**: Same logic as MaxPooling2D — changing a non-maximum pixel doesn't change the output, so its gradient is zero.

🔍 **The loop is explicit**: Unlike `np.add.at` trickery in MaxPooling2D, this implementation uses simple Python loops. It's slower but easier to understand. Each channel of each sample has exactly one argmax position, so there's no risk of overlapping writes.

---

## UpSampling2D

### What does this layer do?

"Nearest neighbor" upsampling: each pixel becomes a block of identical pixels, making the image bigger. If size is `(2, 2)`, every input pixel turns into a 2×2 square of the same value.

### The math

$$y_{i \cdot f_h + p,\; j \cdot f_w + q,\; k} = x_{i,j,k} \quad \text{for } p=0..f_h-1,\; q=0..f_w-1$$

Each input pixel at `(i, j)` is repeated `f_h` times vertically and `f_w` times horizontally.

### Walking through the code

#### `forward`

```python
def forward(self, inputs, training=False):
    self.input_shape_actual = inputs.shape
    return np.repeat(np.repeat(inputs, self.size[0], axis=1), self.size[1], axis=2)
```

🔍 **Line 2**: Cache the input shape. We'll need `H`, `W` and `C` in backward to know how to un-reshape the gradient.

🔍 **Line 3**: `np.repeat` on axis=1 (height) repeats each row `size[0]` times, then `np.repeat` on axis=2 (width) repeats each column `size[1]` times.

📐 **Shape walkthrough**: Input `(B, H, W, C)`. First repeat: `np.repeat(inputs, sh, axis=1)` → `(B, H*sh, W, C)`. Second repeat: `np.repeat(..., sw, axis=2)` → `(B, H*sh, W*sw, C)`.

For example, input `(2, 3, 3, 1)` with size `(2, 2)` → output `(2, 6, 6, 1)`. Pixel `(0, 0)` becomes a 2×2 block of `pixel(0,0)` at positions `(0..1, 0..1)`.

#### `backward`

```python
def backward(self, grad_output):
    batch, h, w, c = self.input_shape_actual
    sh, sw = self.size

    grad = grad_output.reshape(batch, h, sh, w, sw, c)
    return grad.sum(axis=(2, 4))
```

🔍 **Line 2**: Get the original input dimensions from the cache.

🔍 **Line 5**: Reshape the gradient so that the repeated dimensions are separate axes.

📐 **Shape**: `grad_output` is `(B, H*sh, W*sw, C)`. Reshape to `(B, H, sh, W, sw, C)`. Now each original pixel's `sh × sw` block is on axes 2 and 4.

🔍 **Line 6**: Sum over the repeated axes (2 and 4). Since forward repeated the same value across the block, backward sums all those gradients back into a single value.

📐 **Shape**: `(B, H, sh, W, sw, C)` → `.sum(axis=(2,4))` → `(B, H, W, C)` — matching the original input shape.

🔍 **Why sum and not average?** Forward copied each pixel's value `sh * sw` times — it didn't take any kind of average. So if the loss wants to increase a pixel's value, it gets `sh * sw` identical "votes" from the repeated positions. Summing respects the fact that the original pixel's value influences all `sh * sw` output positions equally and independently.

### Try it yourself

```python
from neutro.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, UpSampling2D
import numpy as np

# MaxPooling2D: downsample 28x28 to 14x14
pool = MaxPooling2D(pool_size=(2, 2))
x = np.random.randn(2, 28, 28, 16)
y = pool(x)
print(f"MaxPool: {x.shape} → {y.shape}")         # (2, 28, 28, 16) → (2, 14, 14, 16)

# Global pooling: spatial dims → scalar per channel
gap = GlobalAveragePooling2D()
z = gap(y)
print(f"GlobalAvgPool: {y.shape} → {z.shape}")   # (2, 14, 14, 16) → (2, 16)

gmp = GlobalMaxPooling2D()
z2 = gmp(y)
print(f"GlobalMaxPool: {y.shape} → {z2.shape}")  # (2, 14, 14, 16) → (2, 16)

# UpSampling2D: nearest neighbor upsample
up = UpSampling2D(size=(2, 2))
x_small = np.random.randn(2, 14, 14, 8)
y_big = up(x_small)
print(f"UpSample: {x_small.shape} → {y_big.shape}")  # (2, 14, 14, 8) → (2, 28, 28, 8)
```

## References

- Springenberg, J. T., et al. (2014). **Striving for Simplicity: The All Convolutional Net**. [arXiv:1412.6806](https://arxiv.org/abs/1412.6806)
- Lin, M., Chen, Q., & Yan, S. (2013). **Network In Network**. [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
