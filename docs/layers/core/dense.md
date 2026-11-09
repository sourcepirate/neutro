# Dense Layer

## What does this layer do?

A Dense (fully-connected) layer connects every input feature to every output
feature. It multiplies by a weight matrix, adds a bias, and optionally applies
an activation.

```python
from neutro.layers import Dense
layer = Dense(64, activation="relu")
```

Source: `neutro/layers/core/dense.py`.

---

## The math

$$y = \phi(xW + b)$$

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $x$ | `(B, D)` or `(B, S, D)` | Input; Dense acts on the last axis. |
| $W$ | `(D, U)` | Learned weights. |
| $b$ | `(U,)` | Learned bias, broadcast over the batch. |
| $z = xW + b$ | `(B, U)` | Pre-activation. |
| $\phi$ | element-wise | Optional activation (ReLU, sigmoid, …). |
| $y$ | `(B, U)` | Output. |

For a sequence input `(B, S, D)`, the same `W` is applied at every position:

```text
(B, S, D) @ (D, U) -> (B, S, U)
```

---

## The code

### `__init__`

```python
def __init__(self, units, activation=None, use_bias=True,
             kernel_initializer='glorot_uniform',
             bias_initializer='zeros', **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.activation_name = activation
    self.activation = get_activation(activation)
    self.use_bias = use_bias
    self.kernel_initializer = get_initializer(kernel_initializer)
    self.bias_initializer = get_initializer(bias_initializer)
```

- `units` — number of output features, `U`.
- `activation` — a string like `"relu"` or `None`; resolved to an activation
  object immediately.
- `use_bias` — whether to create `b`.
- The initializers are stored now but run in `build`.

### `build`

```python
def build(self, input_shape):
    self.input_dim = input_shape[-1]
    self.params['W'] = Tensor(self.kernel_initializer((self.input_dim, self.units)))
    if self.use_bias:
        self.params['b'] = Tensor(self.bias_initializer((self.units,)))
    super.build(input_shape)
```

- `input_shape[-1]` is `D`. For `(32, 128)`, `D = 128`.
- `W` is `(D, U)`, `b` is `(U,)`.
- `super.build(input_shape)` sets `self.built = True`.

### `compute_output_shape`

```python
def compute_output_shape(self, input_shape):
    return (*input_shape[:-1], self.units)
```

`(B, D) -> (B, U)`, `(B, S, D) -> (B, S, U)`. Used by `summary` and the
Functional API.

### `forward`

```python
def forward(self, inputs, training=False):
    z = inputs @ self.params['W']
    if self.use_bias:
        z = z + self.params['b']
    if self.activation:
        return self.activation(z)
    return z
```

Shape walkthrough:

```text
inputs: (B, D)
W: (D, U)
z: (B, U)
b: (U,) -> broadcast to (B, U)
y: (B, U)
```

For 3D input, NumPy broadcasts the leading dimensions:

```text
inputs: (B, S, D)
z: (B, S, U)
```

There is no explicit `backward` method. The base `Layer.backward` replays this
forward pass inside a `GradientTape` and fills `self.grads['W']` and
`self.grads['b']`. See [Layer base class](../base.md#backward-the-tape-backed-fallback).

---

## The gradients, for understanding

Even though the tape computes them, the chain-rule formulas are worth knowing:

$$\frac{\partial L}{\partial W} = x^T \frac{\partial L}{\partial z}$$

$$\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial z}$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} W^T$$

If an activation is present, apply its derivative first:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \odot \phi'(z)$$

Shape check for `dW`:

```text
inputs_flat: (B*S, D)
grad_flat: (B*S, U)
dW = inputs_flat.T @ grad_flat -> (D, U) # same shape as W
```

For `db`, sum `grad_flat` over the batch/sequence axes:

```text
db = sum(grad_flat, axis=0) -> (U,)
```

For `dx`, use the original, non-flattened gradient:

```text
grad_output: (B, S, U)
W.T: (U, D)
dx: (B, S, D)
```

---

## Try it yourself

```python
import numpy as np
from neutro.layers import Dense

layer = Dense(units=64, activation="relu")
x = np.random.randn(32, 128)

y = layer(x)
print(y.shape) # (32, 64)
print(layer.count_params()) # 128*64 + 64 = 8256

dL_dy = np.random.randn(32, 64)
dL_dx = layer.backward(dL_dy)
print(np.shape(dL_dx)) # (32, 128)
print(layer.grads['W'].shape) # (128, 64)
print(layer.grads['b'].shape) # (64,)
```

The same layer handles sequences:

```python
x_3d = np.random.randn(8, 10, 128) # (batch, timesteps, features)
y_3d = layer(x_3d)
print(y_3d.shape) # (8, 10, 64)
```

> `Dense.forward` returns a `Tensor`, not a raw NumPy array. Use `.data` when
> you need the underlying array.

---

## What to read next

- [Layer base class](../base.md) — `build`, `forward`, and the tape fallback.
- [GradientTape](../../autograd/tape.md) — the recorder behind the gradient.
- [Tutorial 01](../../tutorials/01-first-model.md) — Dense in a full training loop.
- [Optimizers](../../optimizers/optimizers.md) — what happens after `grads` is filled.
