# Merge Layers: Add, Concatenate, Multiply, Average, Maximum, Minimum

Merge layers combine **multiple input tensors** into a single output tensor. They are essential for building non-linear architectures like ResNets (skip connections), Inception modules, and multi-branch networks. Every merge layer takes a **list of tensors** as input.

## Add — `merging.py:4`

### What does this layer do?

Add computes the element-wise sum of all input tensors. This is the fundamental operation behind **residual (skip) connections**.

### The math, in plain English

$$
y = x_1 + x_2 + \cdots + x_N
$$

Every input must have the **same shape**. The output has that same shape.

### Walking through the code

#### `forward`

```python
def forward(self, inputs, training=False):
    self.input_lengths = len(inputs)
    return sum(inputs)
```

🔍 **Line `self.input_lengths = len(inputs)`**: We cache the number of inputs. The backward pass needs this to know how many gradient tensors to return.

🔍 **Line `return sum(inputs)`**: Python's built-in `sum()` on a list of NumPy arrays performs element-wise addition. All arrays must have the same shape. For example, listing `[a, b, c]` computes `a + b + c`.

📐 **Shape**: If each input is `(batch, 64)`, the output is also `(batch, 64)`.

#### `backward`

```python
def backward(self, grad_output):
    return [grad_output for _ in range(self.input_lengths)]
```

🔍 **Line `[grad_output for _ in range(self.input_lengths)]`**: For $y = x_1 + x_2$, we have $\partial y / \partial x_1 = 1$ and $\partial y / \partial x_2 = 1$. So by the chain rule, $\partial L / \partial x_i = \partial L / \partial y \cdot 1$. The gradient is **broadcast unchanged** to every input. We return a list with `N` identical gradient tensors.

---

## Concatenate — `merging.py:42`

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

## Multiply — `merging.py:85`

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

#### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    res = inputs[0].copy()
    for i in range(1, len(inputs)):
        res *= inputs[i]
    return res
```

🔍 **Line `self.inputs = inputs`**: Cache the list of inputs for the backward pass. The backward pass needs to access all inputs except the one being differentiated.

🔍 **Line `res = inputs[0].copy()`**: Start with a **copy** of the first input. We use `.copy()` to avoid mutating the original input tensor.

🔍 **Line `res *= inputs[i]`**: Multiply element-by-element. After the loop, `res` is the product of all inputs.

📐 **Shape**: `(8, 64)` × `(8, 64)` × `(8, 64)` → `(8, 64)`.

#### `backward`

```python
def backward(self, grad_output):
    grads = []
    for i in range(len(self.inputs)):
        g = grad_output.copy()
        for j in range(len(self.inputs)):
            if i == j:
                continue
            g *= self.inputs[j]
        grads.append(g)
    return grads
```

🔍 **Line `g = grad_output.copy()`**: Start with the upstream gradient.

🔍 **Line `for j ... if i == j: continue; g *= self.inputs[j]`**:
For input $x_i$, we multiply the upstream gradient by **every other input** $x_j$ for $j \neq i$. This implements $\partial L / \partial x_i = \partial L / \partial y \cdot \prod_{j \neq i} x_j$.

📐 **Example with 3 inputs**: $y = a \cdot b \cdot c$.
- $\partial L / \partial a = \partial L / \partial y \cdot b \cdot c$
- $\partial L / \partial b = \partial L / \partial y \cdot a \cdot c$
- $\partial L / \partial c = \partial L / \partial y \cdot a \cdot b$

The loops compute exactly these products.

---

## Average — `merging.py:120`

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
    self.input_lengths = len(inputs)
    return sum(inputs) / self.input_lengths
```

🔍 **Line `self.input_lengths = len(inputs)`**: Cache the number of inputs `N` for the backward pass.

🔍 **Line `sum(inputs) / self.input_lengths`**: Python's `sum()` adds element-wise, then dividing by `N` gives the average.

#### `backward`

```python
def backward(self, grad_output):
    return [grad_output / self.input_lengths for _ in range(self.input_lengths)]
```

🔍 **Line `grad_output / self.input_lengths`**: The derivative of $y = (x_1 + \dots + x_N) / N$ w.r.t. $x_i$ is $1/N$. Each input receives the upstream gradient divided by the number of inputs.

---

## Maximum — `merging.py:144`

### What does this layer do?

Maximum computes the element-wise maximum across all input tensors.

### The math, in plain English

$$
y = \max(x_1, x_2, \dots, x_N)
$$

For each element position, the output is the largest value among all inputs at that position.

The backward pass uses **argmax routing**: the gradient flows only to the input(s) that actually **were** the maximum at each position. All other inputs receive zero gradient.

### Walking through the code

#### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    res = inputs[0].copy()
    for i in range(1, len(inputs)):
        res = np.maximum(res, inputs[i])
    return res
```

🔍 **Line `self.inputs = inputs`**: Cache the inputs. The backward pass needs to compare each input against the maximum.

🔍 **Line `np.maximum(res, inputs[i])`**: Element-wise maximum. `np.maximum(a, b)` returns an array where each element is `max(a_element, b_element)`.

📐 **Shape**: All `(8, 64)`. Output: `(8, 64)`.

#### `backward`

```python
def backward(self, grad_output):
    max_val = self.forward(self.inputs)
    grads = []
    for inp in self.inputs:
        mask = (inp == max_val)
        grads.append(grad_output * mask)
    return grads
```

🔍 **Line `max_val = self.forward(self.inputs)`**: Recompute the maximum values by calling `forward` again. (Alternative: cache `max_val` in forward.)

🔍 **Line `mask = (inp == max_val)`**: For each input, create a boolean mask that is `True` wherever this input equals the maximum value. If multiple inputs share the maximum at a position, all of them get gradient.

🔍 **Line `grad_output * mask`**: The mask zeros out the gradient everywhere this input was **not** the maximum. Only the "winning" input receives gradient.

📐 **The logic**: For $y = \max(x_1, x_2)$, the subgradient is:
$$
\frac{\partial y}{\partial x_1} = \begin{cases} 1 & \text{if } x_1 > x_2 \\ 0 & \text{if } x_1 < x_2 \\ \text{any value in } [0,1] & \text{if } x_1 = x_2 \end{cases}
$$

Neutro uses the tie-case convention: if two inputs are equal, **both** get gradient (the mask is `True` for both).

---

## Minimum — `merging.py:177`

### What does this layer do?

Minimum computes the element-wise minimum across all input tensors. It is the mirror image of Maximum.

### The math, in plain English

$$
y = \min(x_1, x_2, \dots, x_N)
$$

The backward pass uses **argmin routing**: gradient flows only to the input(s) that were the minimum at each position.

### Walking through the code

#### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    res = inputs[0].copy()
    for i in range(1, len(inputs)):
        res = np.minimum(res, inputs[i])
    return res
```

Identical to Maximum but uses `np.minimum`.

#### `backward`

```python
def backward(self, grad_output):
    min_val = self.forward(self.inputs)
    grads = []
    for inp in self.inputs:
        mask = (inp == min_val)
        grads.append(grad_output * mask)
    return grads
```

Identical to Maximum's backward but using the minimum value as the comparison target.

🔍 **Line `mask = (inp == min_val)`**: Gradient passes only where this input equals the minimum. For ties, multiple inputs receive gradient.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition** — skip connections via Add. *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Szegedy, C., et al. (2015). **Going Deeper with Convolutions** — concatenated multi-branch modules. *CVPR*. [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)
