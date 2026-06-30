# `Tensor` — The Autograd Core

## What does it do?

A `Tensor` wraps a NumPy ndarray and optionally records how it was computed. When you call `.backward()`, it walks the computation graph in reverse and fills `.grad` on every tensor that needed a gradient.

## Walkthrough

### `__init__` — What every Tensor holds

```python
def __init__(self, data, requires_grad=False, _children=None, _op=''):
    self.data = np.asarray(data, dtype=float)
    self.grad = None
    self.requires_grad = requires_grad
    self._prev = set(_children) if _children else set()
    self._op = _op
```

🔍 **`self.data`** — The actual NumPy array. This is what gets passed to `np.dot`, `np.add`, etc. during the forward pass. Stored as `float64` for gradient stability.

🔍 **`self.grad`** — Initialized as `None`. After `.backward()`, it holds the gradient of the loss with respect to this tensor, as a NumPy ndarray of the same shape as `self.data`.

🔍 **`self.requires_grad`** — If `False`, the tensor is a "leaf" or "constant" that doesn't need gradients. This prunes the graph: operations on non-grad tensors produce non-grad outputs.

🔍 **`self._prev`** — A `set` of parent Tensors that produced this one. Used by `topological_sort` to traverse the graph. A `set` ensures each parent is visited only once, even if it appears multiple times in one operation (e.g., `a + a`).

🔍 **`self._op`** — A human-readable operation name for debugging: `'add'`, `'matmul'`, `'relu'`, etc.

### `__repr__`

```python
def __repr__(self):
    return f"Tensor(shape={self.data.shape}, grad={'req' if self.requires_grad else 'off'}, op='{self._op}')"
```

Shows shape, gradient requirement, and operation name — enough to trace the graph.

### Properties: `shape`, `ndim`, `T`

```python
@property
def shape(self):
    return self.data.shape

@property
def ndim(self):
    return self.data.ndim

@property
def T(self):
    from .ops import transpose
    return transpose(self)
```

📐 **Shape convenience**: `shape` and `ndim` delegate directly to the underlying NumPy array. `.T` produces a new Tensor with `_op='transpose'`.

### Shape operations

```python
def reshape(self, *shape):
    from .ops import reshape
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return reshape(self, shape)

def transpose(self, *axes):
    from .ops import transpose
    if axes and isinstance(axes[0], (tuple, list)):
        axes = axes[0]
    return transpose(self, axes)
```

These are thin wrappers over `ops.reshape` and `ops.transpose`. They handle flexible Python argument passing (`t.reshape(3, 4)` vs `t.reshape((3, 4))`).

### Reductions

```python
def sum(self, axis=None, keepdims=False):
    from .ops import _sum
    return _sum(self, axis=axis, keepdims=keepdims)

def mean(self, axis=None, keepdims=False):
    from .ops import _mean
    return _mean(self, axis=axis, keepdims=keepdims)
```

### `backward` — The core algorithm

```python
def backward(self, seed=None):
    if seed is not None:
        self.grad = np.asarray(seed, dtype=float)
    else:
        self.grad = np.ones_like(self.data)

    topo = topological_sort(self)
    for node in reversed(topo):
        node._backward()
```

🔍 **Seed the gradient**: If `backward()` is called on a scalar loss, it sets `self.grad = 1.0` (the derivative of the loss w.r.t. itself). You can also pass a custom seed for non-scalar outputs.

🔍 **Topological sort**: `topological_sort(self)` returns all nodes in forward order (leaves first, output last). We iterate in **reverse** — output first — which is the standard reverse-mode AD traversal.

🔍 **Call `_backward`**: Each node's `_backward` closure reads `node.grad` (which was set either by the seed or by a previous node in the reverse traversal), computes gradients for its parents, and adds them to `parent.grad`.

### `zero_grad`

```python
def zero_grad(self):
    self.grad = None
    for child in self._prev:
        child.zero_grad()
```

Resets the gradient on this tensor and recursively on all ancestors. Called between training steps.

### Operator overloads

```python
def __add__(self, other):   return add(self, other)
def __radd__(self, other):  return add(other, self)
def __sub__(self, other):   return sub(self, other)
def __rsub__(self, other):  return sub(other, self)
def __mul__(self, other):   return mul(self, other)
def __rmul__(self, other):  return mul(other, self)
def __truediv__(self, other): return div(self, other)
def __rtruediv__(self, other): return div(other, self)
def __neg__(self):          return neg(self)
def __matmul__(self, other): return matmul(self, other)
def __pow__(self, power):   return _pow(self, power)
def __getitem__(self, idx): return slice_op(self, idx)
```

🔍 **Every operator creates a new Tensor via an `ops` function**. The operator overloads just dispatch to the `ops` module. For example, `a + b` calls `ops.add(a, b)`, which creates a result Tensor with a `_backward` closure.

🔍 **`__radd__`, `__rsub__`, etc. handle `5 + tensor`**. When Python detects the left operand doesn't support the operation, it calls the right operand's reflected method. These dispatch to the same `ops` functions with swapped arguments.

🔍 **`__getitem__` wraps `slice_op`**: Slicing a Tensor creates a new Tensor. The backward closure zeros out a full-size gradient array and writes the upstream gradient into the sliced positions — exactly like PyTorch's `gather` backward.

## Read next

- `docs/autograd/ops.md` — The actual backward rules for every operation
- `docs/autograd/function.md` — The explicit `Function` API for custom ops
