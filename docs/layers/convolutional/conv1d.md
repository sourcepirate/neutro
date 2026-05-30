# Conv1D

## What does this layer do?

Conv1D slides a small 1D filter (kernel) across a sequence, detecting local patterns like phrases in text or short motifs in time-series data. Each filter learns to fire when it sees a specific pattern at a certain position.

## The math, in plain English

$$
Y_{t,f} = \sum_{c=1}^{C} \sum_{k=0}^{K-1} X_{t+k,c} \cdot W_{k,c,f} + b_f
$$

- $X$: input of shape `(B, S, C)` — batch, steps (sequence length), channels.
- $W$: kernel of shape `(K, C, F)` — kernel length, input channels, output filters.
- $b$: bias of shape `(F,)`.
- $Y$: output of shape `(B, S', F)` — same batch, fewer (or same) steps depending on padding & stride, one value per filter.
- $t$: output time step. The filter is centered / aligned at position $t$ in the input.
- $k$: offset within the kernel window (0 through $K-1$).

The core idea: at each position, take a *slice* of the sequence of length $K$, dot it with each of the $F$ filters, and produce one output step. Slide the window by `stride` positions each time.

Padding mode `"valid"` means you never go out of bounds — the output shrinks by $K-1$ steps. `"same"` pads with zeros so the output has the same length as the input (when stride=1).

## The im2col trick

Directly implementing the equation above means nested loops over batch, output steps, input channels, kernel positions, and filters — that is $B \times S' \times C \times K \times F$ iterations in a 6-level loop.

**im2col** (image-to-column) turns this into a single matrix multiply:

1. From the padded input, collect every sliding window of length $K$ over the steps dimension. Each window of shape `(K, C)` is flattened into a column vector of length $K\cdot C$.
2. Stack all these columns side-by-side into a matrix $X_{\text{cols}}$ of shape `(K*C, S')` (ignoring batch, or `(K*C, B*S')` for the batched version).
3. Flatten the kernel $W$ from `(K, C, F)` into a matrix of shape `(F, K*C)`.
4. Compute the output as $W_{\text{flat}} \cdot X_{\text{cols}}$, a single `(F, S')` matrix multiply.

Now convolution is just a matrix multiply — fast, vectorized, and the backward pass is "just transposes."

## Walking through the code

### Step 1: `__init__`

```python
def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None,
             kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,)
    self.strides = strides if isinstance(strides, (tuple, list)) else (strides,)
    self.padding = padding
    self.activation = get_activation(activation)
    self.kernel_initializer = get_initializer(kernel_initializer)
    self.bias_initializer = get_initializer(bias_initializer)
```

🔍 **Line `kernel_size`**: We normalize `kernel_size` to always be a tuple. If the user passes `kernel_size=3`, it becomes `(3,)`. The same is done for `strides` — this avoids guarding against `int` vs `tuple` later.

🔍 **Line `get_activation`**: We support any activation from `neutro.activations` (e.g. `'relu'`, `'sigmoid'`). Turning the name into a callable object is handled by `get_activation()`, which returns either a function or `None` (meaning linear / no activation).

🔍 **Line `kernel_initializer`**: The weight matrix is not created here — only the initializer function is stored. Actual allocation happens in `build()`, which needs to know the input shape.

### Step 2: `build`

```python
def build(self, input_shape):
    _, steps, c = input_shape
    k = self.kernel_size[0]
    self.params['W'] = self.kernel_initializer((k, c, self.filters))
    self.params['b'] = self.bias_initializer((self.filters,))
    super().build(input_shape)
```

📐 **Shape**: kernel `W` is `(K, C, F)` — kernel length, input channels, output filters. Bias `b` is `(F,)` — one scalar per filter, broadcast across batch and steps.

`super().build(input_shape)` sets `self.built = True` and stores `self.input_shape`.

`compute_output_shape` is used by the model for `summary()`:

```python
def compute_output_shape(self, input_shape):
    batch, steps, c = input_shape
    k = self.kernel_size[0]
    s = self.strides[0]
    padding = 0
    if self.padding == 'same':
        padding = (k - 1) // 2
    out_steps = (steps + 2*padding - k) // s + 1
    return (batch, out_steps, self.filters)
```

The formula `(steps + 2*padding - k) // s + 1` is the standard output-length formula for 1D convolution. For `padding='valid'`, `padding=0`, so the window slides exactly `steps - k + 1` times. For `padding='same'`, we add just enough zeros on each side so the output length equals the input length (when `s=1`).

### Step 3: `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    batch, steps, c = inputs.shape
    k = self.kernel_size[0]
    s = self.strides[0]
    f = self.filters

    x = inputs[:, :, None, :].transpose(0, 3, 1, 2)
    W = self.params['W'][:, None, :, :].transpose(3, 2, 0, 1)
    b = self.params['b'].reshape(-1, 1)

    padding = 0
    if self.padding == 'same':
        padding = (k - 1) // 2

    self.x_cols = im2col_indices(x, k, 1, padding=(padding, 0), stride=(s, 1))
    res = W.reshape(f, -1) @ self.x_cols + b

    out_steps = (steps + 2*padding - k) // s + 1

    out = res.reshape(f, out_steps, 1, batch).transpose(3, 1, 2, 0).squeeze(2)
    self.z = out

    if self.activation:
        return self.activation(out)
    return out
```

🔍 **Reshape to 2D for im2col**: Conv1D input is `(B, S, C)`. im2col (from `conv_utils`) expects a 4D tensor shaped `(N, C, H, W)`. So we:

1. `inputs[:, :, None, :]` adds a dummy height dimension: `(B, S, 1, C)`.
2. `.transpose(0, 3, 1, 2)` swaps to channels-first: `(B, C, S, 1)` — batch, channels, steps (height), width=1.

Now `x.shape` is `(B, C, S, 1)` — a 2D image with height = steps and width = 1.

🔍 **Reshape kernel**: `self.params['W']` is `(K, C, F)`. We add a dummy width=1 dimension: `[:, None, :, :]` → `(K, 1, C, F)`. Then `.transpose(3, 2, 0, 1)` → `(F, C, K, 1)`. This matches the format im2col expects.

🔍 **Caching `self.x_cols`**: We save the column matrix. The backward pass needs it to compute `dW`. Caching avoids re-running im2col on the backward pass — a common pattern for memory-vs-speed tradeoff.

📐 **Shape walkthrough**:

- `x_cols` after `im2col_indices(x, k, 1, ...)` has shape `(K*C*1, B * out_steps * 1)` = `(K*C, B * out_steps)`. Each column is one sliding window of length `K` across all `C` channels, flattened.
- `W.reshape(f, -1)` flattens `(F, C, K, 1)` → `(F, K*C)`.
- `res = (F, K*C) @ (K*C, B*out_steps)` → `(F, B*out_steps)`.
- Add bias `b` (reshaped to `(F, 1)` — broadcast).
- `res.reshape(f, out_steps, 1, batch)` → `(F, out_steps, 1, B)`.
- `.transpose(3, 1, 2, 0)` → `(B, out_steps, 1, F)`.
- `.squeeze(2)` → `(B, out_steps, F)` — back to the original shape convention.

🔍 **Why the bias reshape to `(-1, 1)`?** Because `res` is `(F, N)` where `N = B * out_steps`. Adding `b.reshape(-1, 1)` broadcasts `(F, 1)` across all `N` columns, which is the same as adding the bias to each position.

### Step 4: `backward`

```python
def backward(self, grad_output):
    if self.activation:
        if hasattr(self.activation, 'gradient_fast'):
            grad_output = self.activation.gradient_fast(self.z, grad_output)
        else:
            grad_output = grad_output * self.activation.gradient(self.z)

    batch, out_steps, f = grad_output.shape
    k, c, _ = self.params['W'].shape
    s = self.strides[0]

    dout_4d = grad_output[:, :, None, :]
    dout = dout_4d.transpose(3, 1, 2, 0).reshape(f, -1)

    self.grads['b'] = np.sum(grad_output, axis=(0, 1))

    dW = dout @ self.x_cols.T
    self.grads['W'] = dW.reshape(f, c, k, 1).transpose(2, 3, 1, 0).squeeze(1)

    W = self.params['W'][:, None, :, :].transpose(3, 2, 0, 1)
    dx_cols = W.reshape(f, -1).T @ dout

    padding = 0
    if self.padding == 'same':
        padding = (k - 1) // 2

    dx = col2im_indices(dx_cols, (batch, c, self.input_shape[1], 1), k, 1,
                        padding=(padding, 0), stride=(s, 1))
    return dx.transpose(0, 2, 3, 1).squeeze(2)
```

🔍 **Activation gradient**: Before computing layer gradients, we chain through the activation function. If the activation has a `gradient_fast` method, use it (optimized path). Otherwise fall back to the standard `gradient()`. Either way, we multiply element-wise: `dL/dz = dL/dy * dz/da`.

🔍 **Gradient w.r.t. bias**: `grad_output` is `(B, out_steps, F)`. Sum over batch and steps gives `(F,)` — the total gradient for each filter's bias. (Each filter's bias contributes to every position, so we sum all contributions.)

📐 **dW**: `dout` is `(F, B * out_steps)`. `self.x_cols.T` is `(B * out_steps, K*C)`.

`dW = dout @ x_cols.T` → `(F, K*C)`.

Then `dW.reshape(f, c, k, 1).transpose(2, 3, 1, 0).squeeze(1)` reshapes back to `(K, C, F)` — the same shape as the original kernel `W`.

The logic: `dL/dW = x_cols @ dout.T` (or equivalently `dout @ x_cols.T`). This is the matrix-multiply view of the chain rule: the gradient of a matrix multiply `W @ X` w.r.t. `W` is `grad @ X.T`.

📐 **dX through col2im**: We compute `dx_cols = W.T @ dout` — the "transpose convolution" expressed as the transpose of the forward matrix multiply. `W.reshape(f, -1).T` is `(K*C, F)`, `dout` is `(F, N)`, so `dx_cols` is `(K*C, N)`.

Then `col2im_indices` is the **inverse operation of im2col**: it takes each column vector in `dx_cols` and *adds* its elements back to the positions in the original 4D tensor from which they came. When the same input element contributed to multiple output positions (overlapping windows), `np.add.at` (used inside `col2im_indices`) ensures all gradient contributions are **summed**.

The result is `(B, C, S, 1)`, which we transpose to `(B, S, 1, C)` and squeeze back to `(B, S, C)` — the original input shape.

📐 **Gradient shapes**:
| Gradient | Shape | How computed |
|----------|-------|-------------|
| `dL/db` | `(F,)` | `sum(grad_output, axis=(0,1))` |
| `dL/dW` | `(K, C, F)` | `dout @ x_cols.T` reshaped |
| `dL/dX` | `(B, S, C)` | `col2im_indices(W.T @ dout)` |
