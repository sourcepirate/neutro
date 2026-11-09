# `Function` — Custom Ops with an Explicit Backward

Most operations are covered by `Tensor` operator overloads. When you need a
custom operation, subclass `Function`:

```python
class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_data["x"]
        return 2 * x * grad_output
```

Call it with `.apply`:

```python
y = Square.apply(x)
```

Source: `neutro/autograd/function.py`.

---

## What `apply` does

1. Creates a `_Ctx` for saved values.
2. Splits positional and keyword arguments into Tensor inputs and raw data.
3. Calls `forward(ctx, *data, **data_kwargs)` on NumPy arrays.
4. Wraps the output in a `Tensor`.
5. If a tape is active and at least one input is watched, records a backward
   closure on the tape.
6. Returns the result.

The backward closure calls your `backward(ctx, upstream)`, aligns the returned
gradients with the tensor inputs, broadcasts each gradient back to its input
shape, and returns the list to the tape.

---

## `_Ctx`

```python
class _Ctx:
    def __init__(self):
        self.saved_data = {}

    def save_for_backward(self, **kwargs):
        self.saved_data.update(kwargs)
```

Use keyword arguments, not positions:

```python
ctx.save_for_backward(x=x, mask=mask)
x = ctx.saved_data["x"]
```

This is intentionally different from PyTorch's positional
`ctx.save_for_backward(x, mask)`; keyword names make the backward code easier to
read.

Saved values are copied as NumPy arrays, so the backward pass cannot build a
second-order graph through them. That is the educational trade-off.

---

## What `backward` must return

Return one gradient per Tensor input, in the order the tensors appeared:

```python
class AddTwice(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, g):
        return g, g
```

For non-Tensor arguments (ints, floats, shapes), return `None` in that slot if
the argument is positional:

```python
class Scale(Function):
    @staticmethod
    def forward(ctx, x, factor):
        return x * factor

    @staticmethod
    def backward(ctx, g):
        return g *..., None # x grad, factor grad
```

If the returned gradient shape is broadcast, `Function` reduces it back to the
input shape with `utils.broadcast_backward`.

---

## A worked example

```python
import numpy as np
from neutro.autograd import GradientTape, Tensor
from neutro.autograd.function import Function

class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_data["x"]
        return 2 * x * grad_output

x = Tensor(np.array([1.0, 2.0, 3.0]))
with GradientTape as tape:
    tape.watch(x)
    y = Square.apply(x)
    loss = y.sum

tape.gradient(loss, [x])
print(x.grad) # [2. 4. 6.]
```

The derivative of `sum(x**2)` is `2x`, as expected.

---

## When to use `Function` vs plain ops

| Situation | Use |
|-----------|-----|
| Simple arithmetic | `Tensor` operators (`+`, `@`, …) |
| Element-wise activation | `ops.relu`, `ops.softmax`, … |
| Need to save an intermediate | `Function` |
| A non-trivial custom gradient | `Function` |
| Convolution / pooling | Existing `custom_ops` classes |

---

## Read next

- [`GradientTape`](tape.md)
- [`ops`](ops.md)
- [`custom_ops`](autograd.md#architecture)
- [Layer base class](../layers/base.md)
