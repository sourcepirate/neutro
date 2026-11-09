# `ops` — Primitive Operations and Their Backward Rules

## Pattern

Every primitive op is a small `Function` subclass. The public helper wraps it
and returns a `Tensor`:

```python
class _Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a * b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data["a"], ctx.saved_data["b"]
        return g * b, g * a


def mul(a, b):
    return _Mul.apply(a, b)
```

`Function.apply` does the bookkeeping:

1. Split `Tensor` inputs from plain data.
2. Run `forward` on the NumPy arrays.
3. If a tape is active and an input is watched, record a backward closure.
4. Wrap the result in a `Tensor`.

There is no `requires_grad` flag. The tape decides what to record based on
`tape.watch(...)`; see [`GradientTape`](tape.md).

🔍 **Saved values**: `ctx.save_for_backward` stores NumPy arrays, not Tensors.
The backward pass cannot build a second-order graph, which keeps the engine
small.

🔍 **Broadcasting**: `Function.apply` calls `utils.broadcast_backward` on each
returned gradient. This sums over axes that were broadcast in the forward pass,
so a `(B, U)` gradient can become `(U,)` for a bias.

🔍 **Multiple inputs**: `backward` returns one gradient per Tensor input, in the
order the tensors appeared. Non-Tensor arguments get `None` in their slot.

## Arithmetic Operations

### `add(a, b)` → `c = a + b`

**Forward**: `c = a.data + b.data`

**Backward**:
- `dc/da = 1`, so `dL/da = g`
- `dc/db = 1`, so `dL/db = g`

### `sub(a, b)` → `c = a - b`

**Forward**: `c = a.data - b.data`

**Backward**:
- `dc/da = 1`, so `dL/da = g`
- `dc/db = -1`, so `dL/db = -g`

### `mul(a, b)` → `c = a * b`

**Forward**: `c = a.data * b.data`

**Backward**:
- `dc/da = b`, so `dL/da = g * b`
- `dc/db = a`, so `dL/db = g * a`

### `div(a, b)` → `c = a / b`

**Forward**: `c = a.data / b.data`

**Backward**:
- `dc/da = 1/b`, so `dL/da = g / b`
- `dc/db = -a/b²`, so `dL/db = -g * a / b²`

### `neg(x)` → `c = -x`

**Forward**: `c = -x.data`

**Backward**:
- `dc/dx = -1`, so `dL/dx = -g`

### `_pow(x, k)` → `c = x**k` (constant k only)

**Forward**: `c = x.data ** k`

**Backward**:
- `dc/dx = k * x**(k-1)`, so `dL/dx = g * k * x**(k-1)`

### `matmul(a, b)` → `c = a @ b`

**Forward**: `c = a.data @ b.data`

📐 **Shapes**: Let `A` be `(..., M, K)`, `B` be `(..., K, N)`. Then `C = A @ B` is `(..., M, N)`.

**Backward**:
- `dL/dA = g @ B^T` — where `B^T` is `transpose(B, -1, -2)`
- `dL/dB = A^T @ g` — where `A^T` is `transpose(A, -1, -2)`

For 2D:
📐 `(M, K)` = `(M, N) @ (N, K)` → `g @ B.T` gives `(M, N) @ (K, N)` → `(M, K)` ✓
📐 `(K, N)` = `(K, M) @ (M, N)` → `A.T @ g` gives `(K, M) @ (M, N)` → `(K, N)` ✓

For batched matmul, we use `np.swapaxes` on the last two dimensions instead of `.T` (which only reverses all axes).

## Reduction Operations

### `_sum(x, axis=None, keepdims=False)`

**Forward**: `np.sum(x.data, axis)`

**Backward**: Broadcast the gradient back to the original shape. If `axis=None`, the gradient (a scalar) is multiplied by an array of ones matching `x.shape`. If `axis=k`, the gradient is expanded along `axis` to match `x.shape`.

🔍 **Multi-dimensional sum backward**: If `x.shape = (3,4,5)` and `sum(x, axis=1)` reduces to `(3,5)`, then `g` has shape `(3,5)`. We expand: `np.expand_dims(g, axis=1)` → `(3,1,5)`, then broadcast-multiply by ones to get `(3,4,5)`.

### `_mean(x, axis=None, keepdims=False)`

**Forward**: `np.mean(x.data, axis)`

**Backward**: Same as sum but scaled by `1/n` where `n` is the number of elements being averaged.

📐 If `x.shape = (3,4,5)` and `mean(x, axis=1)`: `n = 4` (size of axis 1). `grad_x = g * (1/4)` broadcast to `(3,4,5)`.

## Shape Operations

### `transpose(x, axes=None)`

**Forward**: `np.transpose(x.data, axes)`

**Backward**: `np.transpose(g, inv_axes)` where `inv_axes` reverses the permutation.

🔍 **`inv_axes`**: If `axes = (2, 0, 1)`, then `inv_axes = (1, 2, 0)` — applying `np.argsort` to `(2, 0, 1)` gives `(1, 2, 0)`. Transposing back with the inverse axes recovers the original shape.

### `reshape(x, shape)`

**Forward**: `x.data.reshape(shape)`

**Backward**: `g.reshape(x.shape)` — the upstream gradient is reshaped back to the original input shape.

🔍 **Reshape always works**: Since reshape only changes the view (not the data), `reshape_backward` is just the reverse reshape. No data is lost.

### `slice_op(x, idx)` — Indexing / Slicing

**Forward**: `x.data[idx]`

**Backward**: Create a zero array of `x.shape`, add the gradient into the sliced positions.

🔍 **Zero-fill**: The backward creates a `np.zeros_like(x.data)` and adds `g` into `full[idx]`. Using `+=` accumulation handles the case where multiple slices write to the same position (admittedly rare in practice).

## Activation Functions

### `relu(x)`

**Forward**: `np.maximum(0, x.data)`

**Backward**: `g * (x.data > 0)` — zero out gradients for negative inputs.

🔍 **ReLU dies at 0**: The gradient is 0 at exactly 0 (standard convention). In practice this is fine — hitting exactly 0 is measure-zero.

### `sigmoid(x)`

**Forward**: `1 / (1 + exp(-x))`

**Backward**: `g * s * (1 - s)` where `s = sigmoid(x)`.

📐 **Derivative**: `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))`. The backward recomputes `s` from `x.data` in the closure.

### `tanh(x)`

**Forward**: `np.tanh(x.data)`

**Backward**: `g * (1 - t²)` where `t = tanh(x)`.

📐 **Derivative**: `tanh'(x) = 1 - tanh²(x)`. Recomputes `t` in the closure.

### `silu(x)` — Sigmoid Linear Unit

**Forward**: `x * sigmoid(x)`

**Backward**: `g * (s + x * s * (1 - s))` where `s = sigmoid(x)`.

📐 **Derivative**: `silu'(x) = s + x * s' = s + x * s * (1 - s)`. Derived from the product rule.

🔍 **SiLU vs Swish**: SiLU is equivalent to the Swish activation (`x * sigmoid(x)`), popularized by the Swish paper (Ramachandran et al., 2017) and used in Llama / GPT-4.

### `softmax(x, axis=-1)`

**Forward**: `exp(x - max) / sum(exp(x - max))` (stable softmax)

**Backward**: `s * (g - sum(s * g, axis, keepdims=True))`

📐 **Jacobian**: The softmax Jacobian is not diagonal: `J_ij = s_i * (δ_ij - s_j)`. The full matrix-vector product `J @ g` is: `s_i * (g_i - Σ_j s_j * g_j)` for each `i`. This is computed as `s * (g - dot(s, g))` — vectorized across the batch.

🔍 **Why not element-wise sigmoid gradient?**: Using `s * (1 - s)` for softmax backward gives incorrect gradients because it ignores the cross-terms (`i ≠ j`). The correct formula accounts for the fact that increasing one output necessarily decreases the others (since outputs sum to 1).

## Read next

- [GradientTape](tape.md) — how recorded ops are replayed backward
- [Function](function.md) — custom ops with explicit forward/backward
- [Layer base class](../layers/base.md) — how layers use the tape
