# KerasTensor, Node, and the Functional API Graph Engine

The Functional API builds models as directed acyclic graphs (DAGs) instead of
linear stacks. Before real data exists, the model needs a way to track symbolic
data flow.

Source: `neutro/engine/node.py`.

---

## The two objects

### `KerasTensor`

```python
class KerasTensor:
    def __init__(self, shape, node=None, name=None):
        self.shape = shape
        self.node = node
        self.name = name
```

- `shape` is a tuple like `(None, 32)`; the batch axis is `None` until runtime.
- `node` points back to the `Node` that produced this tensor.
- It carries no data.

### `Node`

```python
class Node:
    def __init__(self, layer, input_tensors, output_tensors):
        self.layer = layer
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        if not hasattr(layer, '_inbound_nodes'):
            layer._inbound_nodes = []
        layer._inbound_nodes.append(self)
        ...
```

A `Node` records one **call** to a layer:

- `input_tensors` -> layer -> `output_tensors`.
- It registers itself on `layer._inbound_nodes`.
- Multi-output layers store a list in `output_tensors`.
- Shared layers get one `Node` entry per call.

`_inbound_nodes` is how `Model` discovers graph connections for `summary()`.

---

## How symbolic calls happen

`Layer.__call__` checks whether the input is symbolic:

```python
def __call__(self, inputs, *args, **kwargs):
    if self._is_symbolic_input(inputs):
        return self._symbolic_call(inputs)
    return self._eager_call(inputs, args, kwargs)
```

`_symbolic_call` does **no computation**:

```python
input_shapes = [i.shape for i in inputs] if isinstance(inputs, list) else inputs.shape
if not self.built:
    self.build(input_shapes)
output_shape = self.compute_output_shape(input_shapes)
output_tensors = KerasTensor(shape=output_shape)
Node(self, input_tensors=inputs, output_tensors=output_tensors)
return output_tensors
```

It builds the layer if needed, infers the output shape, creates a new
`KerasTensor`, and records the connection.

---

## Graph discovery

`Model._init_graph` walks backward from the output tensors:

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

This is a post-order DFS. `nodes_ordered` ends up in forward execution order;
`Model.backward` iterates `reversed(nodes_ordered)`.

`Model` also collects unique layers from the ordered nodes, which is what
`summary()` and the optimizer traverse.

---

## Forward execution

```python
tensor_map = {}
tensor_map[id(model.inputs)] = inputs

for node in model._nodes_ordered:
    if isinstance(node.layer, InputLayer):
        continue
    node_inputs = tensor_map.get(id(node.input_tensors))
    output = node.layer.forward(node_inputs, training=training)
    node.layer._last_inputs = as_tensor(node_inputs)
    node.layer._last_kwargs = {'training': training}
    node.state = model._capture_layer_state(node.layer)
    tensor_map[id(node.output_tensors)] = output
```

- `InputLayer` nodes are skipped; user data is seeded into `tensor_map`.
- Each node captures a deep copy of its layer state after forward. This is how
  a shared layer can appear in multiple branches and still backward correctly.

---

## Backward execution

`Model.backward` runs the nodes in reverse:

1. Look up the gradient for this node's outputs.
2. Restore the captured state for this call.
3. Clear the layer's grads.
4. Call `node.layer.backward(grad_output)`.
5. Accumulate parameter gradients for shared layers.
6. Add input gradients to the graph's `grad_map`.

The layer's `backward` is usually the base tape-backed implementation; see
[Layer base class](../layers/base.md#backward-the-tape-backed-fallback).

---

## Read next

- [Model and Sequential](../models/model.md)
- [Layer base class](../layers/base.md)
- [InputLayer](../layers/core/input_layer.md)
- [Architecture map](../ARCHITECTURE.md)
