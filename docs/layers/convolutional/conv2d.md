# Conv2D

## What does this layer do?

Conv2D slides a 2D filter bank across an image (or any 2D grid), detecting spatial patterns like edges, textures, and shapes. Each filter learns to respond when a particular visual pattern appears in its receptive field.

## The math, in plain English

$$
Y_{h',w',f} = \sum_{c=1}^{C} \sum_{i=0}^{K_H-1} \sum_{j=0}^{K_W-1} X_{h'+i,\,w'+j,\,c} \cdot W_{i,j,c,f} + b_f
$$

- $X$: input of shape `(B, H, W, C)` — batch, height, width, channels.
- $W$: kernel of shape `(KH, KW, C, F)` — kernel height, kernel width, input channels, output filters.
- $b$: bias of shape `(F,)`.
- $Y$: output of shape `(B, OH, OW, F)` — batch, output height, output width, filters.
- $(h', w')$: output spatial position. The filter aligns with input position $(h' \cdot s_H, w' \cdot s_W)$.
- $i, j$: offsets within the kernel window.

The formula is a **2D cross-correlation** (conventionally called convolution in deep learning): at each output position, you take a `(KH, KW)` patch of the input, multiply it element-wise by each filter in the bank, sum everything up, and add a bias.

## Padding

For `padding='valid'`: no padding. Output shrinks: `OH = (H - KH) // stride_H + 1`.

For `padding='same'`: pad with zeros so the output has the same spatial size as input (when stride=1). Each side gets `(KH - 1) // 2` and `(KW - 1) // 2` zeros.

## The im2col trick

Convolution is expensive if you write it as nested loops (6-7 levels deep). **im2col** unrolls each sliding window into a column of a big matrix:

1. Pad the input (if needed).
2. For every `(KH, KW)` window position, take all `C` channels, flatten into a column vector of length `KH * KW * C`.
3. Stack all columns: $X_{\text{cols}}$ is `(KH*KW*C, N)` where `N = B * OH * OW`.
4. Flatten kernel to `(F, KH*KW*C)`.
5. Convolution = one matrix multiply: `W_flat @ X_cols` → `(F, N)`, then reshape to `(B, OH, OW, F)`.

Now convolution is a **single BLAS-level matrix multiply** — and the backward pass falls out for free via transposes.

## data_format: `channels_last` vs `channels_first`

```python
self.data_format = data_format
```

Conv2D supports both conventions:

- `channels_last` (default): `(B, H, W, C)` — TensorFlow convention.
- `channels_first`: `(B, C, H, W)` — PyTorch convention.

Three helper methods normalize everything to `channels_last` internally:

```python
_shape_to_channels_last(shape)
_to_channels_last(inputs)     # convert input to NHWC
_from_channels_last(outputs)  # convert output back to original format
```

This keeps the core convolution logic (and im2col) in one format while exposing the user's preferred convention at the API boundary.

## Walking through the code

### Step 1: `__init__`

```python
def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None,
             kernel_initializer='glorot_uniform', bias_initializer='zeros',
             data_format='channels_last', **kwargs):
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    self.padding = padding
    self.activation = get_activation(activation)
    self.kernel_initializer = get_initializer(kernel_initializer)
    self.bias_initializer = get_initializer(bias_initializer)
    if data_format not in ('channels_last', 'channels_first'):
        raise ValueError("data_format must be 'channels_last' or 'channels_first'")
    self.data_format = data_format
```

🔍 **Line `kernel_size`**: If the user passes an integer (e.g., `kernel_size=3`), we normalize to `(3, 3)`. Same for `strides`. This means the rest of the code can always unpack two values without checking types.

🔍 **Line `data_format`**: Keras-style `data_format` argument. The validation check at line 30 catches typos early rather than producing mysterious shape errors later.

### Step 2: `build`

```python
def build(self, input_shape):
    _, h, w, c = self._shape_to_channels_last(input_shape)
    kh, kw = self.kernel_size
    self.params['W'] = self.kernel_initializer((kh, kw, c, self.filters))
    self.params['b'] = self.bias_initializer((self.filters,))
    super().build(input_shape)
```

📐 **Shape**: kernel `W` is `(KH, KW, C, F)` — kernel height, kernel width, input channels, output filters. Bias `b` is `(F,)`.

Note that `_shape_to_channels_last` converts the input shape to NHWC if it's in `channels_first`, so `build` always sees the canonical shape.

`compute_output_shape` mirrors the output length formula for 2D:

```python
oh = (h + 2*padding - kh) // sh + 1
ow = (w + 2*padding - kw) // sw + 1
```

### Step 3: `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    inputs_nhwc = self._to_channels_last(inputs)
    batch, h, w, c = inputs_nhwc.shape
    kh, kw, _, f = self.params['W'].shape
    sh, sw = self.strides

    x = inputs_nhwc.transpose(0, 3, 1, 2)
    W = self.params['W'].transpose(3, 2, 0, 1)
    b = self.params['b'].reshape(-1, 1)

    padding = 0
    if self.padding == 'same':
        padding = (kh - 1) // 2

    self.x_cols = im2col_indices(x, kh, kw, padding=padding, stride=sh)
    res = W.reshape(f, -1) @ self.x_cols + b

    oh = (h + 2*padding - kh) // sh + 1
    ow = (w + 2*padding - kw) // sw + 1

    out = res.reshape(f, oh, ow, batch).transpose(3, 1, 2, 0)
    self.z = out

    if self.activation:
        out = self.activation(out)
    return self._from_channels_last(out)
```

🔍 **Convert to NHWC**: `_to_channels_last` ensures the forward pass always works with `(B, H, W, C)` regardless of the user's `data_format`. If already NHWC, it's a no-op.

🔍 **Prepare for im2col**: im2col expects `(N, C, H, W)` — standard PyTorch format. So we transpose `(B, H, W, C)` → `(B, C, H, W)`.

The kernel `W` is originally `(KH, KW, C, F)`. We transpose to `(F, C, KH, KW)` to match the im2col channel ordering. After reshaping to `(F, -1)`, each row is one filter flattened across all `KH * KW * C` elements.

🔍 **The big multiply**: `W.reshape(f, -1)` → `(F, KH*KW*C)`. `self.x_cols` from `im2col_indices` → `(KH*KW*C, N)` where `N = B * OH * OW`.

`res = (F, KH*KW*C) @ (KH*KW*C, N)` → `(F, N)`.

📐 **Shape walkthrough** for the output reshape:

- `res` is `(F, N)` where `N = B * OH * OW`.
- `.reshape(f, oh, ow, batch)` → `(F, OH, OW, B)`.
- `.transpose(3, 1, 2, 0)` → `(B, OH, OW, F)` — the final NHWC output.

🔍 **Caching `self.x_cols`**: Just like in Conv1D, we save the column matrix because the backward pass needs `x_cols` to compute `dW`. This is the time-vs-memory tradeoff typical in training loops: pay memory cost per layer to avoid recomputing im2col on every backward pass.

### Step 4: `backward`

```python
def backward(self, grad_output):
    grad_output_nhwc = self._to_channels_last(grad_output)
    if self.activation:
        if hasattr(self.activation, 'gradient_fast'):
            grad_output_nhwc = self.activation.gradient_fast(self.z, grad_output_nhwc)
        else:
            grad_output_nhwc = grad_output_nhwc * self.activation.gradient(self.z)

    batch, oh, ow, f = grad_output_nhwc.shape
    kh, kw, c, _ = self.params['W'].shape
    sh, sw = self.strides

    dout = grad_output_nhwc.transpose(3, 1, 2, 0).reshape(f, -1)

    self.grads['b'] = np.sum(grad_output_nhwc, axis=(0, 1, 2))

    dW = dout @ self.x_cols.T
    self.grads['W'] = dW.reshape(f, c, kh, kw).transpose(2, 3, 1, 0)

    W = self.params['W'].transpose(3, 2, 0, 1)
    dx_cols = W.reshape(f, -1).T @ dout

    padding = 0
    if self.padding == 'same':
        padding = (kh - 1) // 2

    _, h, w, _ = self._shape_to_channels_last(self.input_shape)
    dx = col2im_indices(dx_cols, (batch, c, h, w), kh, kw, padding=padding, stride=sh)
    return self._from_channels_last(dx.transpose(0, 2, 3, 1))
```

🔍 **Convert grad_output**: The incoming gradient might be `channels_first` — we normalize it to NHWC first so the shapes align with the cached `x_cols`.

🔍 **Activation chain rule**: Same pattern as Conv1D — `gradient_fast` if available, otherwise element-wise multiply by the activation gradient. This computes `dL/dz` from `dL/dy`.

🔍 **dW**: `dout` is `(F, N)` (flattened NHWC grad). `self.x_cols.T` is `(N, KH*KW*C)`.

`dW = dout @ x_cols.T` → `(F, KH*KW*C)`.

`.reshape(f, c, kh, kw)` → `(F, C, KH, KW)`. `.transpose(2, 3, 1, 0)` → `(KH, KW, C, F)` — back to the original kernel shape.

This is the same as Conv1D: the gradient of `W @ X` w.r.t. `W` is `grad @ X^T`, and then we reshape back to the parameter's native shape.

🔍 **dX via col2im**: We compute the gradient through the matrix multiply: `dx_cols = W^T @ dout` where `W.reshape(f, -1).T` is `(KH*KW*C, F)` and `dout` is `(F, N)`. So `dx_cols` is `(KH*KW*C, N)`.

Now we call `col2im_indices` — the **reverse** of `im2col_indices`:

```
dx = col2im_indices(dx_cols, (batch, c, h, w), kh, kw, padding=padding, stride=sh)
```

🔍 **How col2im works**: `im2col_indices` extracted overlapping patches from the input and laid them out as columns of a matrix. `col2im_indices` does the reverse: it takes each column, *scatters* its elements back to their original `(N, C, H, W)` positions, and when multiple patches overlap at the same position, **sums** the overlapping gradient contributions (using `np.add.at`).

This is exactly the adjoint / transpose of the im2col operation — the gradient of an unrolling operation is the "re-rolling" operation that adds contributions back.

📐 **Final output**: `dx` is `(B, C, H, W)` (NCHW). We transpose to `(B, H, W, C)` (NHWC) and then convert back to the user's `data_format` via `_from_channels_last`.

📐 **Gradient shapes**:
| Gradient | Shape | How computed |
|----------|-------|-------------|
| `dL/db` | `(F,)` | `sum(grad_output, axis=(0,1,2))` |
| `dL/dW` | `(KH, KW, C, F)` | `dout @ x_cols.T` reshaped |
| `dL/dX` | `(B, H, W, C)` (or NCHW) | `col2im_indices(W.T @ dout)` |
