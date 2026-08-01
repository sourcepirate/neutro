# Merge Layers: Add, Concatenate, Multiply, Average, Maximum, Minimum

Merge layers combine **multiple input tensors** into a single output tensor. They are essential for building non-linear architectures like ResNets (skip connections), Inception modules, and multi-branch networks. Every merge layer takes a **list of tensors** as input.

## Shared structure

All reduce-type merge layers (`Add`, `Multiply`, `Average`, `Maximum`, `Minimum`) inherit from `_ReduceBase` (`merging.py:11`), which provides the common `build`, `compute_output_shape`, and a `forward` that folds a single binary operation over the input list. Each concrete layer only implements `_combine(left, right)` — the per-pair reduction:

```python
def forward(self, inputs, training=False):
    if not isinstance(inputs, list):
        return inputs
    result = _to_tensor(inputs[0])
    for other in inputs[1:]:
        result = self._combine(result, _to_tensor(other))
    return result
```

`_to_tensor` coerces plain NumPy inputs into autograd `Tensor`s so the whole reduction is differentiable.

## Add — `merging.py:42`

### What does this layer do?

Add computes the element-wise sum of all input tensors. This is the fundamental operation behind **residual (skip) connections**.

### The math, in plain English

$$
y = x_1 + x_2 + \cdots + x_N
$$

Every input must have the **same shape**. The output has that same shape.

### Walking through the code

#### `_combine`

```python
def _combine(self, left, right):
    return left + right
```

📐 **Shape**: If each input is `(batch, 64)`, the output is also `(batch, 64)`.

The additive reduction routes the gradient unchanged back to every input, so skip connections receive clean gradient flow.

---

## Concatenate — `merging.py:47`

### What does this layer do?

Concatenate joins multiple tensors along a specified axis. All inputs must have the same shape **except** along the concatenation axis, where their dimensions are summed. This is the core of multi-branch feature fusion architectures like Inception.

### The math, in plain English

$$
y = [x_1, x_2, \dots, x_N] \quad \text{along axis } a
$$

If each input has shape $(d_0, d_1, \dots, d_a, \dots, d_k)$ and we concatenate along axis $a$, the output has shape $(d_0, d_1, \dots, \sum_i d_a^{(i)}, \dots, d_k)$.

### Walking through the code

#### `__init__`

```python
def __init__(self, axis=-1, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis
```

🔍 **`axis=-1`**: By default, concatenation happens along the last axis (features). This is the most common use case — joining feature vectors side-by-side.

#### `compute_output_shape`

```python
def compute_output_shape(self, input_shape):
    if not isinstance(input_shape, list):
        return input_shape

    out_shape = list(input_shape[0])
    concat_dim = 0
    for shape in input_shape:
        dim = shape[self.axis]
        if dim is None:
            concat_dim = None
            break
        concat_dim += dim

    out_shape[self.axis] = concat_dim
    return tuple(out_shape)
```

🔍 **Line `if dim is None: concat_dim = None; break`**: This handles **symbolic shapes** where the batch dimension (or any dimension) is `None` at graph-building time. If any input has a `None` dimension on the concat axis, the output's concat dimension will also be `None`.

📐 **Example**: Input shapes `[(None, 10), (None, 20)]` with `axis=-1` → `out_shape = (None, 30)`. But if one dimension is `None` on the concat axis, it propagates as `None`.

#### `forward`

```python
def forward(self, inputs, training=False):
    self.input_shapes = [i.shape for i in inputs]
    return np.concatenate(inputs, axis=self.axis)
```

🔍 **Line `self.input_shapes = [i.shape for i in inputs]`**: We cache the **actual shape** of each input tensor. The backward pass needs the sizes along the concat axis to split the gradient correctly.

🔍 **Line `np.concatenate(inputs, axis=self.axis)`**: NumPy's native concatenation. This is the only operation — no learned parameters.

📐 **Shape**: `[a, b, c]` where `a.shape = (8, 10)`, `b.shape = (8, 20)`, `c.shape = (8, 30)` with `axis=-1` → `(8, 60)`.

#### `backward`

```python
def backward(self, grad_output):
    indices = np.cumsum([s[self.axis] for s in self.input_shapes])[:-1]
    return np.split(grad_output, indices, axis=self.axis)
```

🔍 **Line `indices = np.cumsum(...)`**: Compute the split points from the cached input shapes. `np.cumsum` gives cumulative sums along the concat axis. We drop the last element with `[:-1]` because `np.split` takes split positions.

📐 **Example**: Input shapes along axis: `[10, 20, 30]`. `np.cumsum([10, 20, 30])` = `[10, 30, 60]`. `[:-1]` = `[10, 30]`. These are the split indices: slice 0..10, 10..30, 30..60.

🔍 **Line `np.split(grad_output, indices, axis=self.axis)`**: Reverse of `np.concatenate`. Splits the gradient along the same axis into the original pieces. Returns a list of gradient tensors matching the input shapes.

---

## Multiply — `merging.py:79`

### What does this layer do?

Multiply computes the element-wise product of all input tensors. This is useful in attention mechanisms, gating, and specialized architectures.

### The math, in plain English

$$
y = x_1 \odot x_2 \odot \cdots \odot x_N
$$

Where $\odot$ denotes element-wise multiplication. All inputs must have the same shape.

For the backward pass, the gradient w.r.t. a single input is the product of **all other inputs** times the upstream gradient:

$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y} \odot \prod_{j \neq i} x_j
$$

### Walking through the code

#### `_combine`

```python
def _combine(self, left, right):
    return left * right
```

The base `_ReduceBase.forward` folds this pairwise product over all inputs: `(a * b) * c`.

📐 **Shape**: `(8, 64)` × `(8, 64)` × `(8, 64)` → `(8, 64)`.

#### `backward`

The `Multiply` reduction is performed on autograd `Tensor`s, so the reverse-mode engine (`GradientTape`) computes the gradients automatically. For $y = x_1 \odot x_2 \odot \cdots \odot x_N$:

$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y} \odot \prod_{j \neq i} x_j
$$

📐 **Example with 3 inputs**: $y = a \cdot b \cdot c$.
- $\partial L / \partial a = \partial L / \partial y \cdot b \cdot c$
- $\partial L / \partial b = \partial L / \partial y \cdot a \cdot c$
- $\partial L / \partial c = \partial L / \partial y \cdot a \cdot b$

The autograd engine computes exactly these products.

---

## Average — `merging.py:84`

### What does this layer do?

Average computes the element-wise mean of all input tensors.

### The math, in plain English

$$
y = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

### Walking through the code

#### `forward`

```python
def forward(self, inputs, training=False):
    result = super().forward(inputs, training=training)   # sum via _combine
    if isinstance(inputs, list) and len(inputs) > 1:
        result = result / float(len(inputs))
    return result
```

The base `_ReduceBase.forward` sums all inputs with `_combine(a, b) = a + b`, then `Average` divides by the number of inputs `N`.

#### `backward`

The derivative of $y = (x_1 + \dots + x_N) / N$ w.r.t. $x_i$ is $1/N$. Each input receives the upstream gradient divided by the number of inputs, computed automatically by the autograd engine.

---

## Maximum — `merging.py:95`

### What does this layer do?

Maximum computes the element-wise maximum across all input tensors.

### The math, in plain English

$$
y = \max(x_1, x_2, \dots, x_N)
$$

For each element position, the output is the largest value among all inputs at that position.

The backward pass uses **argmax routing**: the gradient flows only to the input(s) that actually **were** the maximum at each position. All other inputs receive zero gradient.

### Walking through the code

#### `_combine`

```python
def _combine(self, left, right):
    return autograd_ops.maximum(left, right)
```

The base `_ReduceBase.forward` folds this pairwise `maximum` over all inputs. `autograd_ops.maximum` is differentiable, so the backward pass runs through the autograd engine.

📐 **Shape**: All `(8, 64)`. Output: `(8, 64)`.

#### `backward`

The autograd engine uses **argmax routing**: the gradient flows only to the input(s) that actually **were** the maximum at each position. All other inputs receive zero gradient.

🔍 **The logic**: For $y = \max(x_1, x_2)$, the subgradient is:
$$
\frac{\partial y}{\partial x_1} = \begin{cases} 1 & \text{if } x_1 > x_2 \\ 0 & \text{if } x_1 < x_2 \\ \text{any value in } [0,1] & \text{if } x_1 = x_2 \end{cases}
$$

Neutro uses the tie-case convention: if two inputs are equal, **both** get gradient.

---

## Minimum — `merging.py:100`

### What does this layer do?

Minimum computes the element-wise minimum across all input tensors. It is the mirror image of Maximum.

### The math, in plain English

$$
y = \min(x_1, x_2, \dots, x_N)
$$

The backward pass uses **argmin routing**: gradient flows only to the input(s) that were the minimum at each position.

### Walking through the code

#### `_combine`

```python
def _combine(self, left, right):
    return -autograd_ops.maximum(-left, -right)
```

`Minimum` is implemented as the negation of `Maximum`: `min(a, b) = -max(-a, -b)`. This reuses the differentiable `maximum` op.

#### `backward`

Identical to Maximum's backward: gradient passes only where this input equals the minimum (argmin routing), computed by the autograd engine. For ties, multiple inputs receive gradient.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition** — skip connections via Add. *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Szegedy, C., et al. (2015). **Going Deeper with Convolutions** — concatenated multi-branch modules. *CVPR*. [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)
