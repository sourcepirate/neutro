# `GradientTape` — Recording Operations

`GradientTape` is the recorder behind Neutro's automatic differentiation. It
does not build a permanent graph. It records the operations that run inside a
`with` block, then walks them in reverse when you ask for gradients.

Source: `neutro/autograd/tape.py`.

---

## A minimal example

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

`tape.watch(W)` is required. The tape records an op only if at least one of its
inputs is watched (`Function.apply` checks `_should_record`).

---

## The pieces

### `_tape_stack`

A module-level list of active tapes. `get_active_tape` returns the top one.
`__enter__` pushes, `__exit__` pops. This is why ops do not need a tape
argument: `Function.apply` asks the global stack whether a tape is recording.

### `watch(tensor)`

Adds a `Tensor` to `self._watched`. It accepts a single tensor or an iterable.
Non-tensors raise `TypeError`; this catches the common mistake of watching a
NumPy array.

### `_record_op(inputs, output, backward_fn, name)`

Appends a record:

```python
{
    "inputs": [...], # the Tensor inputs
    "output": output, # the result Tensor
    "backward": fn, # callable(upstream) -> list of input grads
    "name": "MatMul", # for error messages
}
```

It also marks the output as watched so downstream ops are recorded.

### `gradient(target, sources)`

1. Seed the gradient table with `d target / d target = 1`.
2. Walk the recorded ops in reverse.
3. For each op whose output has a gradient, call its backward function with
   the upstream gradient.
4. Accumulate the returned input gradients (out-of-place addition).
5. Write the final gradient for each source into `source.grad` and return the
   list.

If a source has no path to the target, its gradient is `None`.

---

## What it deliberately does not do

- **No higher-order gradients.** Gradients are NumPy arrays, not Tensors.
- **No in-place op tracking.** Mutating `.data` directly bypasses the tape.
- **No persistent graph.** Once `gradient` runs, the records are still there,
  but the intended pattern is one tape per forward/backward step.
- **No `requires_grad` flag.** You explicitly `watch` what you want.

The last point is the biggest difference from PyTorch. A `Tensor` is just data
plus a gradient slot; the tape decides what is differentiable.

---

## Where it is used

- `Layer.backward` (`neutro/layers/base.py`) replays `forward` inside a tape.
- `Model.autograd_fit` wraps the whole forward + loss in one tape.
- `Function.apply` (`neutro/autograd/function.py`) records custom ops.

Read next: [`Tensor`](tensor.md), [`Function`](function.md), [Autograd engine](autograd.md).
