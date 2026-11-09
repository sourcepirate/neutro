# `Tensor` — Data Plus Gradient

`Tensor` is a thin wrapper around a NumPy array. It holds two things:

- `data` — the numbers.
- `grad` — the gradient of the loss with respect to this tensor, or `None`.

It does **not** have `requires_grad`, `_prev`, `_op`, or `.backward()`. Those
belong to the tape. This is the biggest difference from PyTorch-style tensors.

Source: `neutro/autograd/tensor.py`.

---

## Creating a tensor

```python
import numpy as np
from neutro.autograd import Tensor

a = Tensor([1.0, 2.0, 3.0])
print(a.data) # [1. 2. 3.]
print(a.grad) # None
print(a.shape) # (3,)
print(a.dtype) # float64
```

`Tensor` converts input to `float64`. If you pass another `Tensor`,
`as_tensor` returns it unchanged; if you pass a list or tuple, it converts each
item.

---

## The attributes

| Attribute | Meaning |
|-----------|---------|
| `data` | NumPy `float64` array. |
| `grad` | NumPy array of the same shape, or `None`. |
| `shape`, `ndim`, `size`, `dtype` | Delegated to `data`. |
| `T` | Transpose (a new `Tensor`). |

`zero_grad` sets `grad` back to `None`. `copy` copies the data but not the
gradient.

---

## Arithmetic builds a tape

```python
b = a * 2 + 1
```

`a * 2` calls `Tensor.__mul__`, which dispatches to `ops.mul`; `+ 1` dispatches
to `ops.add`. If a tape is active and `a` is watched, each op records a
backward rule. The result is a new `Tensor`, and the tape remembers how to get
back to `a`.

`Tensor` implements `+`, `-`, `*`, `/`, `@`, `**`, `-x`, and the reflected
versions (`2 + a`, `2 * a`, …). It also implements `reshape`, `transpose`,
`sum`, `mean`, `sqrt`, and `pow`.

---

## Indexing and mutation

```python
x = Tensor(np.arange(6).reshape(2, 3))
row = x[0] # differentiable slicing
x[0, 0] = 99.0 # in-place mutation; clears x.grad
```

`__getitem__` uses `ops.slice_op`, so slicing is differentiable. `__setitem__`
mutates `data` directly and sets `grad = None`; in-place mutation is not tracked
by the tape.

---

## Comparisons return NumPy arrays

```python
a = Tensor([1.0, 2.0])
print(a > 1.0) # [False True] — a NumPy array, not a Tensor
```

This is deliberate: comparisons are for masks and debugging, not for
differentiation. Because `__eq__` returns an array, `Tensor.__hash__` is
identity-based so tensors can still be used as dictionary keys.

---

## A complete example

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

Notice:

- `W.grad` is a NumPy array.
- `x` is not watched, so no gradient is computed for it.
- The tape, not the tensor, owns the computation history.

---

## What to read next

- [`GradientTape`](tape.md) — the recorder and reverse pass.
- [`ops`](ops.md) — the backward rule for each operation.
- [`Function`](function.md) — custom ops.
- [Layer base class](../layers/base.md) — how layers use the tape.
