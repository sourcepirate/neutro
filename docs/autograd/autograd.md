# `neutro.autograd` — A NumPy Autograd Engine

## What is this?

`neutro.autograd` is an **educational reverse-mode automatic differentiation engine** built entirely from NumPy, styled after PyTorch's `autograd`.

It is **additive and opt-in** — the existing manual `backward()` layers in `neutro` are untouched. This subpackage exists to show **how an autograd engine works under the hood**.

## Quickstart

```python
from neutro.autograd import Tensor
import numpy as np

a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
c = a * b + b
loss = c.sum()
loss.backward()

print(a.grad)  # [4. 5. 6.]   — d(loss)/da
print(b.grad)  # [5. 6. 7.]   — d(loss)/db  (a*da/db + db/db = a + 1)
```

## Architecture

| Module | File | Purpose |
|--------|------|---------|
| `Tensor` | `tensor.py` | Data + gradient + backward closure |
| `ops` | `ops.py` | Primitive arithmetic, reductions, activations |
| `Function` | `function.py` | Explicit forward/backward op API |
| `utils` | `utils.py` | Topological sort, broadcast backward, grad helpers |
| `adapter` | `adapter.py` | Bridge to existing `Layer.params/grads` |

## How it works

1. **Define-by-run**: Every arithmetic operation (`+`, `*`, `@`, etc.) builds a `Tensor` and attaches a `_backward` closure that captures its parents. The graph is the trace of execution.

2. **Topological backward**: `Tensor.backward()` topologically sorts the graph from the loss back to the leaves, then calls each `_backward` closure. Closures write `gr.ad` into `Tensor.grad` (accumulating for shared nodes).

3. **No autograd-through-autograd**: Gradients are NumPy arrays, not Tensors. There is no higher-order AD. This keeps the implementation clear and educational.

## What's covered (v1)

The first version supports core math + linear algebra:

- **Arithmetic**: `add`, `sub`, `mul`, `div`, `neg`, `pow` (constant), `matmul`
- **Reductions**: `sum`, `mean` (with axis)
- **Shape ops**: `reshape`, `transpose`, `slice`
- **Broadcasting**: all binary ops handle broadcasting with correct backward reductions
- **Activations**: `relu`, `sigmoid`, `tanh`, `silu`, `softmax`
- **Custom ops**: `Function` base class for defining explicit forward/backward ops

Not (yet) covered: convolution, pooling, normalization, LSTM steps.

## File-by-file

- `neutro/autograd/tensor.py` — The core `Tensor` class (read the walkthrough)
- `neutro/autograd/ops.py` — Every primitive operation and its backward rule
- `neutro/autograd/function.py` — `Function` base class for custom explicit ops
- `neutro/autograd/utils.py` — Topological sort, broadcast backward, numeric grad check
- `neutro/autograd/adapter.py` — Bridging autograd `Tensor` into existing `Layer.params/grads`

## Diff from PyTorch

| Aspect | PyTorch | neutro.autograd |
|--------|---------|-----------------|
| Backend | ATen/C++ | NumPy |
| Grad type | Tensor | NumPy ndarray |
| Higher-order AD | Yes | No |
| In-place ops | Yes | No |
| Device support | CPU/CUDA/... | CPU only |

The absence of higher-order AD and in-place ops is intentional — it keeps the implementation under ~300 lines and focused on the core algorithm.

## Read next

- `docs/autograd/tensor.md` — How `Tensor` works, line-by-line
- `docs/autograd/ops.md` — Every operation's math and backward rule
- `docs/autograd/function.md` — The `Function` base class for custom ops
- `docs/autograd/adapter.md` — Bridging into existing `neutro` layers
