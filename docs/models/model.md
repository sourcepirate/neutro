# `Model`, `Sequential`, and the Functional API

Neutro supports three ways to build a model, mirroring Keras.

Source files:

- `neutro/models/base_model.py` — `Model` and `Sequential` definitions.
- `neutro/models/training.py` — the epoch/batch loop, one-step update, and
  validation evaluation.
- `neutro/models/summary.py` — the `summary` table.

---

## 1. Sequential API

A linear stack of layers. Each layer's output feeds the next.

```python
from neutro.layers import Dense
from neutro.models import Sequential

model = Sequential([
    Dense(16, activation="relu", input_shape=(8,)),
    Dense(3, activation="softmax"),
])
```

`Sequential.add` builds each layer from the previous layer's output shape. If
the first layer did not receive `input_shape`, the first real call builds it.

Forward: iterate `self.layers` in order.
Backward: iterate in reverse and call each layer's `backward`.

---

## 2. Functional API

A graph (DAG) where any output can feed any other layer. Use `Input` and call
layers on the symbolic tensor:

```python
from neutro.layers import Add, Dense, Input
from neutro.models import Model

inputs = Input(shape=(8,))
x = Dense(16, activation="relu")(inputs)
residual = Dense(16, activation="relu")(x)
merged = Add([x, residual])
outputs = Dense(3, activation="softmax")(merged)

model = Model(inputs=inputs, outputs=outputs)
```

Key objects:

- `KerasTensor` — a symbolic placeholder with a shape (`neutro/engine/node.py`).
- `Node` — records one layer call: input tensors -> output tensors.
- `Model._init_graph` — traverses backward from outputs, topologically sorts
  the nodes, and collects unique layers.

---

## 3. Subclassed API

Subclass `Model` and write `forward`/`backward` yourself. Used by `UNet`, `VAE`,
and `DiffusionModel`.

```python
from neutro.models import Model

class MyModel(Model):
    def __init__(self):
        super().__init__
        self.fc = Dense(3)

    def forward(self, x, training=False):
        return self.fc(x, training=training)
```

If your `forward` uses `Tensor` operations, the base `Model.backward` still
works for a Sequential-style layer list. Custom graphs can override
`backward` (see `VAE` and `UNet`).

---

## Graph discovery (`_init_graph`)

```python
def traverse(tensor):
    if hasattr(tensor, 'node') and tensor.node:
        node = tensor.node
        if node not in visited_nodes:
            visited_nodes.add(node)
            for t in node.input_tensors:
                traverse(t)
            nodes_ordered.append(node)
```

This is a post-order DFS from the outputs. The result, `_nodes_ordered`, is in
forward execution order; backward iterates `reversed(_nodes_ordered)`.

---

## Functional forward

```python
tensor_map = {}
tensor_map[id(self.inputs)] = inputs

for node in self._nodes_ordered:
    if isinstance(node.layer, InputLayer):
        continue
    node_inputs = tensor_map.get(id(node.input_tensors))
    output = node.layer.forward(node_inputs, training=training)
    node.layer._last_inputs = as_tensor(node_inputs)
    node.layer._last_kwargs = {'training': training}
    node.state = self._capture_layer_state(node.layer)
    tensor_map[id(node.output_tensors)] = output
```

Two details matter:

- `InputLayer` nodes are skipped; real data is seeded in `tensor_map`.
- Each node captures a deep copy of its layer state **after** forward. This is
  what lets a shared layer be called in two branches and still backward
  correctly.

---

## Functional backward

```python
grad_map = {}
grad_map[id(self.outputs)] = grad
layer_grads_accumulator = {}

for node in reversed(self._nodes_ordered):
    node_grad_outputs = grad_map.get(id(node.output_tensors))
    if node_grad_outputs is None:
        continue

    self._restore_layer_state(node.layer, node.state)
    self._clear_layer_grads(node.layer)
    grad_inputs = node.layer.backward(node_grad_outputs)
    self._accumulate_layer_grads(node.layer, layer_grads_accumulator)

    # add grad_inputs to grad_map for node.input_tensors
```

- **State restore** gives each node the intermediate values from its own
  forward call.
- **Gradient accumulation** sums parameter gradients when a layer is reused.
- **Branch accumulation** adds input gradients when one tensor feeds several
  downstream nodes.

---

## `fit` vs `autograd_fit`

Both use the same `_fit_loop` and the same `History`/callbacks.

| Method | Gradient path |
|--------|---------------|
| `model.fit(...)` | Calls each layer's `backward`. The base `Layer.backward` is tape-backed; manual layers override it. |
| `model.autograd_fit(...)` | One `GradientTape` wraps the whole forward pass and loss; `tape.gradient` fills `Tensor.grad`. |

`fit` is the default and the beginner path. `autograd_fit` is useful for seeing
the entire computation as a single graph.

### The training loop

`_fit_loop` supports:

1. **Single array**: `fit(x, y)`.
2. **MIMO**: `fit([x1, x2], [y1, y2])`.
3. **Generator**: an object with `__len__`/`__iter__` and optionally
   `batch_size`; each iteration yields `(x_batch, y_batch)`.

For each batch:

```text
forward -> loss -> backward -> optimizer.step -> metrics
```

Callbacks receive `on_train_begin`, `on_epoch_begin`, `on_batch_begin`,
`on_batch_end`, `on_epoch_end`, and `on_train_end`.

---

## `think` about shapes

`Model.compute_output_shape` walks `self.layers` for Sequential models and
returns the recorded symbolic shapes for Functional models. `summary` uses
it, along with `count_params`, to print the table.

---

## Save and load

```python
model.save("model.joblib")
restored = Model.load("model.joblib")
```

This uses `joblib` and serializes the whole Python object. It is convenient for
experiments, not a portable production format.

---

## Read next

- [Layer base class](../layers/base.md)
- [Dense layer](../layers/core/dense.md)
- [GradientTape](../autograd/tape.md)
- [Tutorial 02](../tutorials/02-gradients-and-the-tape.md)
