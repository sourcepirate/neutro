# `Function` — Custom Ops with Explicit Backward

## Why?

While most operations are covered by the Tensor operator overloads, sometimes you need a custom operation with a specific backward rule. The `Function` base class provides a PyTorch-style API: define `forward` and `backward` as static methods, use `ctx.save_for_backward` to stash values, then call `MyOp.apply(args)`.

## Walkthrough

### `_Ctx` — The context object

```python
class _Ctx:
    def __init__(self):
        self.saved_data = {}

    def save_for_backward(self, **kwargs):
        self.saved_data.update(kwargs)
```

🔍 **`save_for_backward`** — Saves NumPy arrays (not Tensors) needed during backward. In PyTorch, `ctx.save_for_backward` can save tensors and the autograd engine can differentiate through backward. Here, we deliberately save raw data to keep it educational — no second-order gradients.

🔍 **`**kwargs` style**: Unlike PyTorch's positional API, we use keyword arguments. This avoids confusion about order and makes the backward code self-documenting: `ctx.saved_data['x']` vs `ctx.saved_tensors[0]`.

### `Function.apply` — The core glue

```python
@classmethod
def apply(cls, *args, **kwargs):
    ctx = _Ctx()

    tensor_args = []
    data_args = []
    for a in args:
        if isinstance(a, Tensor):
            tensor_args.append(a)
            data_args.append(a.data)
        else:
            data_args.append(a)
    ...
```

🔍 **Separate tensors from scalars**: We iterate over all args. Tensors get their `.data` extracted and stored separately; non-Tensor args (integers, floats, ndarrays) pass through as-is. The `tensor_args` list is used later to distribute gradients.

```python
    output_data = cls.forward(ctx, *data_args, **data_kwargs)

    requires_grad = any(getattr(a, 'requires_grad', False) for a in tensor_args)
    if not requires_grad:
        return Tensor(output_data, requires_grad=False)
```

🔍 **Prune the graph**: If no input tensor requires gradients, the output is a plain Tensor with `requires_grad=False` — no graph overhead.

```python
    result = Tensor(
        output_data,
        requires_grad=True,
        _children=tensor_args,
        _op=cls.__name__,
    )
```

🔍 **`_children=tensor_args`**: We pass the list of tensor args as children. Since `_children` is passed as a positional arg to `Tensor.__init__`, it becomes `self._prev = set(_children)` — deduplicating for the topological sort.

```python
    def bw():
        grad_data = result.grad
        if grad_data is None:
            return
        grad_inputs = cls.backward(ctx, grad_data)
        if not isinstance(grad_inputs, (list, tuple)):
            grad_inputs = [grad_inputs]
        for t, g in zip(tensor_args, grad_inputs):
            if t.requires_grad and g is not None:
                _acc_grad(t, broadcast_backward(g, t.shape))

    result._backward = bw
    return result
```

🔍 **The `_backward` closure**: When `result.backward()` runs, `result.grad` is already set (by the topological sort). The closure:
1. Reads `result.grad` (the upstream gradient).
2. Calls `cls.backward(ctx, grad_data)` to get gradients for each input.
3. Wraps a single gradient in a list for uniform handling.
4. For each tensor arg that requires grad, accumulates the gradient (handling broadcasting).

## Example: `Square`

```python
class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_data['x']
        return 2 * x * grad_output
```

🔍 **`forward(ctx, x)`**: Takes the context and any number of positional args. Saves the input for backward, returns the squared output. **No Tensors here** — `x` is a NumPy array.

🔍 **`backward(ctx, grad_output)`**: Retrieves the saved `x`, applies the chain rule: `d/dx(x**2) = 2x`, multiplies by the upstream gradient. The returned gradient must match the shape of the corresponding input.

## When to use `Function` vs raw ops

| Use case | API |
|----------|-----|
| Simple arithmetic | Operator overloads (implicit) |
| Activation functions | `ops.relu`, `ops.softmax` (implicit) |
| Need to save intermediate values | `Function` (explicit) |
| Complex custom gradient | `Function` (explicit) |

## Read next

- `docs/autograd/adapter.md` — Using autograd with existing `neutro` layers and optimizers
