# Layer Base Class

Every layer in `neutro` inherits from `neutro.layers.base.Layer`. The base class
defines the **layer lifecycle** and the two execution modes:

- **eager mode** — real data flows through `forward`.
- **symbolic mode** — the Functional API builds a graph using `KerasTensor`.

It also provides a generic, tape-backed `backward` for layers that only write
`forward`.

Source: `neutro/layers/base.py`.

---

## The contract

A layer may implement:

| Method | Required? | Purpose |
|--------|-----------|---------|
| `__init__` | Yes | Store hyperparameters; call `super().__init__`. |
| `build(input_shape)` | Usually | Create parameters in `self.params`; call `super.build`. |
| `forward(inputs, training=False)` | Yes | The actual computation. |
| `compute_output_shape(input_shape)` | Recommended | Used by `summary` and the Functional API. |
| `backward(grad_output)` | Optional | Override only for manual backprop. The base version uses the tape. |

---

## `__init__` — the fields every layer has

```python
class Layer:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.trainable = True
        self.built = False
        self.params = {}
        self.grads = {}
        self.input_shape = kwargs.get('input_shape')
        self.output_shape = None
        self._inbound_nodes = []
```

- `name` — label for `summary` and debugging.
- `trainable` — the optimizer skips a layer when this is `False`.
- `built` — `False` until `build` has created parameters.
- `params` — `dict[str, Tensor]`, e.g. `{"W":..., "b":...}`.
- `grads` — `dict[str, np.ndarray]`, filled by `backward`.
- `input_shape` — stored for `summary` and shape inference.
- `_inbound_nodes` — Functional API bookkeeping.

### `sublayers`

```python
@property
def sublayers(self):...
```

Walks every attribute of the layer, including lists, tuples, and dicts, and
returns all nested `Layer` objects once. This is how a `TransformerBlock` exposes
its attention, norms, and feed-forward layers to the optimizer.

### `count_params`

```python
def count_params(self):
    own = sum(p.data.size if isinstance(p, AutoTensor) else p.size
              for p in self.params.values())
    return own + sum(layer.count_params() for layer in self.sublayers)
```

Counts parameters owned by this layer plus every nested layer. `summary`
uses this.

### `compute_output_shape`

```python
def compute_output_shape(self, input_shape):
    return self.output_shape if self.output_shape is not None else input_shape
```

The default assumes the shape is unchanged. Layers that change the last
dimension (like `Dense`) or spatial dimensions (like `Conv2D`) override it.

---

## `__call__` — the dispatch hub

```python
def __call__(self, inputs, *args, **kwargs):
    if self._is_symbolic_input(inputs):
        return self._symbolic_call(inputs)
    return self._eager_call(inputs, args, kwargs)
```

If the input contains a `KerasTensor`, we are building a graph. Otherwise we
are running real data.

### Eager mode

```python
def _eager_call(self, inputs, args, kwargs):
    t_inputs = _as_tensor(inputs)
    if not self.built:
        self.build(t_inputs.shape)
    self._last_inputs = t_inputs
    self._last_args = args
    self._last_kwargs = kwargs
    output = self.forward(t_inputs, *args, **kwargs)
    self._last_output = output
    return output
```

Three things to notice:

1. Inputs are converted to `Tensor`. A plain NumPy array becomes a `Tensor`
   automatically.
2. `build` runs once, on the first call, using the real input shape.
3. The inputs, positional args, and keyword args are cached. The base
   `backward` uses them to replay `forward`.

### Symbolic mode

```python
def _symbolic_call(self, inputs):
    input_shapes = [i.shape for i in inputs] if isinstance(inputs, list) else inputs.shape
    if not self.built:
        self.build(input_shapes)
    output_shape = self.compute_output_shape(input_shapes)
    output_tensors = KerasTensor(shape=output_shape)
    Node(self, input_tensors=inputs, output_tensors=output_tensors)
    return output_tensors
```

No numbers flow. The layer records a `Node` connecting input tensors to output
tensors; `Model` later walks those nodes to run the graph.

---

## `backward` — the tape-backed fallback

Most layers do not implement `backward`. The base version does this:

```python
def backward(self, grad_output):
    t_inputs = self._captured_inputs()
    with GradientTape() as tape:
        for source in all_sources:
            tape.watch(source)
        output = self.forward(t_inputs, *fwd_args, **fwd_kwargs)
        loss = (output * Tensor(grad_output)).sum

    tape.gradient(loss, all_sources)
    # copy Tensor.grad into layer.grads for every sublayer
    return t_inputs.grad
```

In plain English:

1. Take the inputs and runtime args from the last forward call.
2. Collect every `Tensor` parameter in this layer and its sublayers.
3. Re-run `forward` with those inputs and parameters watched.
4. Build the scalar probe `(output * grad_output).sum`.
5. Ask the tape for gradients with respect to every source.
6. Copy the parameter gradients into `layer.grads` and return the input
   gradient.

**Why the probe works:** if `L` is the true scalar loss and `g = dL/doutput`,
then `d/dp [(output * g).sum] = dL/dp` for every parameter `p`. The probe has
the same gradient as the original loss but is a scalar the tape can
differentiate directly.

**What this means for you:** a layer whose `forward` uses only `Tensor`
operations gets a correct backward pass for free. You only write `backward`
when the explicit chain rule is the lesson (see
[Implementation status](../IMPLEMENTATION_STATUS.md#backward-strategies)).

---

## Manual backprop (advanced)

If you do override `backward`, the rules are the ordinary chain rule. For
`y = x @ W + b`:

```text
dL/dW = x.T @ dL/dy
dL/db = sum(dL/dy, axis=0)
dL/dx = dL/dy @ W.T
```

For rank-3 inputs, flatten the leading dimensions before the weight gradient
and reshape the input gradient back:

```text
x: (B, S, D) -> (B*S, D)
dy: (B, S, U) -> (B*S, U)
dW = x_flat.T @ dy_flat # (D, U)
```

Manual layers must cache whatever `backward` needs (`self.inputs`, `self.z`,
masks, etc.) during `forward`.

---

## Try it yourself: a layer with no `backward`

```python
import numpy as np
from neutro.layers.base import Layer
from neutro.autograd import Tensor

class MyDense(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        rng = np.random.default_rng(0)
        self.params['W'] = Tensor(rng.normal(0, 0.1, (input_shape[-1], self.units)))
        self.params['b'] = Tensor(np.zeros(self.units))
        super.build(input_shape)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def forward(self, inputs, training=False):
        return inputs @ self.params['W'] + self.params['b']

layer = MyDense(3)
x = np.random.randn(4, 2)
y = layer(x)

print(y.shape) # (4, 3)
print(layer.count_params()) # 2*3 + 3 = 9

g = layer.backward(np.ones((4, 3)))
print(np.shape(g)) # (4, 2)
print(layer.grads['W'].shape) # (2, 3)
print(layer.grads['b'].shape) # (3,)
```

Notice: `MyDense` never defines `backward`, yet `layer.grads` is filled. That is
the tape fallback doing its job.

---

## What to read next

- [Dense layer](core/dense.md) — a concrete layer with shapes and gradients.
- [GradientTape](../autograd/tape.md) — how the recorder works.
- [Model and Sequential](../models/model.md) — how layers are trained.
- [Architecture map](../ARCHITECTURE.md) — how a call flows through the system.
