# `neutro.autograd` — A NumPy Autograd Engine

## What is this?

`neutro.autograd` is an **educational reverse-mode automatic differentiation
engine** built entirely from NumPy. It is the engine behind `Layer.backward`
and `Model.autograd_fit`.

It is small on purpose. The goal is to show the machinery, not to compete with
PyTorch.

Source: `neutro/autograd/`.

---

## Quickstart

```python
import numpy as np
from neutro.autograd import GradientTape, Tensor

x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
W = Tensor(np.array([[0.5], [-0.5]]))

with GradientTape as tape:
    tape.watch(W)
    loss = ((x @ W) ** 2).mean

tape.gradient(loss, [W])
print(float(loss.data)) # 0.25
print(W.grad) # [[-2.]
                           # [-3.]]
```

The tape records operations whose inputs are watched, then walks them in reverse
to fill `.grad`.

---

## Architecture

| Module | Purpose |
|--------|---------|
| `tensor.py` | `Tensor`: a NumPy array plus a `grad` slot. |
| `tape.py` | `GradientTape`: records ops and computes reverse-mode gradients. |
| `ops.py` | Primitive operations (`add`, `matmul`, `softmax`, …) and their backward rules. |
| `function.py` | `Function`: define a custom op with an explicit forward/backward pair. |
| `custom_ops.py` | Higher-level `Function`s used by layers (conv, pooling, gather, quantize). |
| `utils.py` | Broadcast backward, numeric gradient checks. |

There is no `adapter.py` in the current version. The tape is the integration
point: layers write `forward` with `Tensor` operations and the base
`Layer.backward` uses the tape.

---

## How it works

1. **Define-by-run.** Every arithmetic operation on a `Tensor` calls a function
   in `ops.py`, which creates a new `Tensor` and records a backward rule on the
   active tape.
2. **Watch what you differentiate.** Unlike PyTorch, `Tensor` has no
   `requires_grad` flag. You call `tape.watch(tensor)` explicitly. If no input
   to an op is watched, the op is not recorded.
3. **Reverse pass.** `tape.gradient(target, sources)` seeds
   `d target / d target = 1`, walks the records in reverse, calls each
   backward rule with the upstream gradient, and accumulates the results.
4. **No higher-order AD.** Gradients are NumPy arrays, not Tensors. This keeps
   the implementation readable.

---

## What is covered

- **Arithmetic**: `add`, `sub`, `mul`, `div`, `neg`, `pow`, `matmul`
- **Reductions**: `sum`, `mean`, `max`
- **Shape ops**: `reshape`, `transpose`, `slice`, `concatenate`
- **Activations**: `relu`, `sigmoid`, `tanh`, `silu`, `softmax`
- **Broadcasting**: binary ops reduce gradients back to the original shape
- **Custom ops**: `Function` subclasses (convolution, pooling, gather, quantize)
- **Numeric checks**: `numeric_grad`, `gradclose` in `utils.py`

Not covered: higher-order gradients, in-place op tracking, GPU execution, and
autograd through the tape's own bookkeeping.

---

## Two ways to use it

### 1. Through layers (the normal path)

```python
model.fit(x, y, epochs=1)
```

Each layer's base `backward` replays `forward` inside a tape. You do not write
gradient code for `Dense`, `Conv2D`, attention, and so on.

### 2. Through `Model.autograd_fit` (the whole-model path)

```python
model.autograd_fit(x, y, epochs=1)
```

One tape wraps the entire forward pass and loss. This is useful when you want to
see the whole computation as a single graph.

Both are built on the same `GradientTape`.

---

## Read next

- [`Tensor`](tensor.md) — data + gradient.
- [`GradientTape`](tape.md) — record and reverse.
- [`ops`](ops.md) — every primitive and its backward rule.
- [`Function`](function.md) — custom ops.
- [Tutorial 02](../tutorials/02-gradients-and-the-tape.md) — a guided walkthrough.
