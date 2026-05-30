# Merge Layers: Add, Concatenate, Multiply, Average, Maximum, Minimum

## Theory

Merge layers combine multiple input tensors into a single output tensor. They are essential for building non-linear architectures like ResNets (skip connections), Inception modules, and multi-branch networks. Every merge layer takes a **list of tensors** as input.

### Operations

| Layer | Operation | Gradient |
|---|---|---|
| `Add` | $y = \sum_i x_i$ | $\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y}$ (same for all) |
| `Multiply` | $y = \prod_i x_i$ | $\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y} \odot \prod_{j \ne i} x_j$ |
| `Average` | $y = \frac{1}{N} \sum_i x_i$ | $\frac{\partial L}{\partial x_i} = \frac{1}{N} \frac{\partial L}{\partial y}$ |
| `Maximum` | $y = \max_i x_i$ | $\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y} \odot \mathbf{1}_{x_i = y}$ |
| `Minimum` | $y = \min_i x_i$ | $\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y} \odot \mathbf{1}_{x_i = y}$ |
| `Concatenate` | $y = [x_1, x_2, \dots, x_N]$ along axis $a$ | Split $y$ gradient back along $a$ |

For `Maximum`/`Minimum`, the indicator function $\mathbf{1}_{x_i = y}$ passes the gradient only to the input(s) that achieved the extreme value — this is known as **argmax routing** in gradient computation.

## Implementation Guide

### File: `neutro/layers/core/merging.py`

### `Add` — line 4

```python
class Add(Layer):
    def forward(self, inputs, training=False):
        self.input_lengths = len(inputs)
        return sum(inputs)

    def backward(self, grad_output):
        return [grad_output for _ in range(self.input_lengths)]
```

- `sum(inputs)` works element-wise across the list.
- The gradient is **broadcast unchanged** to every input — the sum's Jacobian w.r.t. each input is the identity.

### `Concatenate` — line 37

```python
class Concatenate(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        out_shape = list(input_shape[0])
        concat_dim = 0
        for shape in input_shape:
            dim = shape[self.axis]
            if dim is None:    # Handle symbolic None (batch dim)
                concat_dim = None
                break
            concat_dim += dim
        out_shape[self.axis] = concat_dim
        return tuple(out_shape)

    def forward(self, inputs, training=False):
        self.input_shapes = [i.shape for i in inputs]
        return np.concatenate(inputs, axis=self.axis)

    def backward(self, grad_output):
        indices = np.cumsum([s[self.axis] for s in self.input_shapes])[:-1]
        return np.split(grad_output, indices, axis=self.axis)
```

- `compute_output_shape` correctly handles symbolic `None` dimensions (e.g., batch size).
- The backward uses `np.split` to reverse the concatenation along the same axis.

### `Multiply` — line 74

```python
class Multiply(Layer):
    def forward(self, inputs, training=False):
        self.inputs = inputs
        res = inputs[0].copy()
        for i in range(1, len(inputs)):
            res *= inputs[i]
        return res

    def backward(self, grad_output):
        grads = []
        for i in range(len(self.inputs)):
            g = grad_output.copy()
            for j in range(len(self.inputs)):
                if i == j: continue
                g *= self.inputs[j]   # Product of all inputs except the i-th
            grads.append(g)
        return grads
```

- For each input $i$, the gradient is $\frac{\partial L}{\partial y} \odot \prod_{j \ne i} x_j$.
- `self.inputs` is cached during forward for use in backward (important for shared layer state restoration).

### `Average`, `Maximum`, `Minimum` — lines 127-200

These follow the same pattern. `Maximum` and `Minimum` use `np.maximum` / `np.minimum` in forward and mask-based gradient routing in backward.

### Shared Layer Compatibility

All merge layers store intermediate state (`input_lengths`, `input_shapes`, `inputs`) on `self` during `forward`. For shared merge layers used multiple times in a graph, the `Model` class uses `_capture_layer_state` / `_restore_layer_state` (recursive, covering sublayers) to save and restore this state per node.

## Usage Example

```python
from neutro.layers import Input, Dense, Add, Concatenate
from neutro.models import Model

# Skip connection (Add)
inp = Input(shape=(32,))
x = Dense(32, activation='relu')(inp)
skip = Dense(32)(x)
out = Add()([x, skip])  # Two branches merged

# Multi-branch concatenation
i1 = Input(shape=(10,))
i2 = Input(shape=(20,))
merged = Concatenate(axis=-1)([i1, i2])  # Output shape: (None, 30)
```

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition** — skip connections via Add. *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Szegedy, C., et al. (2015). **Going Deeper with Convolutions** — concatenated multi-branch modules. *CVPR*. [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)
