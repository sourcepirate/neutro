# Tutorial 05 — Manual Backprop (Advanced)

**Time:** about 20 minutes. **Prerequisites:** [Tutorial 02](02-gradients-and-the-tape.md).

The beginner path uses the tape: write `forward`, let `Layer.backward` compute
gradients. This lesson is the other path: write the chain rule by hand.

Run the companion script:

```bash
python examples/05_manual_backprop.py
```

Expected output:

```text
Forward difference:       0.0
dW difference:            0.0
db difference:            0.0
dInput difference:        0.0
```

The manual and tape-backed layers agree exactly on this example.

---

## 1. The formulas

For a Dense layer:

```text
y = x @ W + b
```

Given the upstream gradient `g = dL/dy`:

```text
dL/dW = x.T @ g
dL/db = sum(g, axis=0)
dL/dx = g @ W.T
```

Shape check:

```text
x:  (N, D)
W:  (D, U)
g:  (N, U)

x.T @ g:  (D, N) @ (N, U) -> (D, U)   # same shape as W
sum(g):   (U,)                        # same shape as b
g @ W.T:  (N, U) @ (U, D) -> (N, D)   # same shape as x
```

If an activation is present, first multiply `g` by the activation derivative
evaluated at the pre-activation:

```text
g_z = g * activation'(z)
```

Then use `g_z` in the three formulas.

---

## 2. A manual layer

```python
from neutro.autograd import Tensor
from neutro.layers.base import Layer
import numpy as np


class ManualDense(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        rng = np.random.default_rng(0)
        self.params["W"] = Tensor(rng.normal(0, 0.1, (input_shape[-1], self.units)))
        self.params["b"] = Tensor(np.zeros(self.units))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def forward(self, inputs, training=False):
        x = inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs)
        self._x = x
        z = x @ self.params["W"].data + self.params["b"].data
        return Tensor(z)

    def backward(self, grad_output):
        g = grad_output.data if isinstance(grad_output, Tensor) else np.asarray(grad_output)
        self.grads["W"] = self._x.T @ g
        self.grads["b"] = g.sum(axis=0)
        return g @ self.params["W"].data.T
```

Why each cache exists:

- `self._x` is stored in `forward` because `dW` needs the input. The tape path
  does not need this cache because it replays `forward`.
- `self.params["W"]` is a `Tensor`, so the optimizer can update it.
- `backward` writes NumPy arrays into `self.grads`; that is the contract the
  optimizer reads.

---

## 3. Compare with the tape

```python
manual = ManualDense(4)
auto = Dense(4)
manual.build((3, 5))
auto.build((3, 5))

# Same weights.
auto.params["W"].data = manual.params["W"].data.copy()
auto.params["b"].data = manual.params["b"].data.copy()

y_manual = manual(x)
y_auto = auto(x)

dx_manual = manual.backward(upstream)
dx_auto = auto.backward(upstream)
```

Then compare:

```python
print(np.max(np.abs(manual.grads["W"] - auto.grads["W"])))
print(np.max(np.abs(manual.grads["b"] - auto.grads["b"])))
print(np.max(np.abs(dx_manual - dx_auto)))
```

All three should be `0.0` (or within floating-point noise).

---

## 4. When to write it by hand

The tape is the default because it is hard to get wrong. Manual backprop is
worth it when:

- the gradient derivation is the lesson;
- a layer has a branched structure that is awkward to replay;
- you need to control exactly what is cached.

In Neutro, manual backprop is used by:

- `BitLinear`
- `LoRADense` and `QLoRADense`
- `UNet`
- `VAE`
- `MaskedDiffusionLM`

Every other layer uses the tape-backed fallback. See
[Implementation status](../IMPLEMENTATION_STATUS.md#backward-strategies).

---

## 5. Common mistakes

1. **Forgetting the activation derivative.** If `forward` applies ReLU, the
   gradient must pass through `(z > 0)` first.
2. **Wrong transpose.** `x.T @ g` is `(D, U)`; `g @ x.T` is not.
3. **Not summing the bias gradient.** The bias is shared by every sample.
4. **Forgetting to flatten leading dimensions.** For `(B, S, D)`, flatten to
   `(B*S, D)` for the weight gradient, then reshape `dx` back.
5. **Writing into a parameter instead of its gradient.** `optimizer.step`
   expects the derivative in `self.grads`, not an already-updated parameter.
6. **Caching a view that later changes.** Copy or keep the data stable.

---

## 6. Your turn

1. Add a ReLU to `ManualDense.forward` and apply `(z > 0)` in `backward`.
   Compare with `Dense(units, activation="relu")`.
2. Add a bias argument. Does `db` still sum over axis 0 for a 3D input?
3. Implement `ManualLayerNorm` and compare it with `LayerNormalization`.
4. Use `utils.numeric_grad` to finite-difference your manual gradient and
   check it.

<details>
<summary>Hint for the 3D case</summary>

For `x` of shape `(B, S, D)`, flatten the leading axes:

```text
x_flat: (B*S, D)
g_flat: (B*S, U)
dW = x_flat.T @ g_flat       # (D, U)
db = g_flat.sum(axis=0)      # (U,)
dX = (g_flat @ W.T).reshape(B, S, D)
```

</details>

---

## What to read next

- [Layer base class](../layers/base.md)
- [GradientTape](../autograd/tape.md)
- [Dense layer](../layers/core/dense.md)
- [Implementation status](../IMPLEMENTATION_STATUS.md#backward-strategies)
- [Tutorial 03 — Attention from scratch](03-attention-from-scratch.md)
