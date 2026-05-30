# KerasTensor, Node, and the Functional API Graph Engine

## Theory

The Functional API lets you build models as directed acyclic graphs (DAGs) of layers, rather than as linear stacks. This requires a mechanism to track *symbolic* data flow during model construction, before any real data is seen.

Two core classes enable this:

- **`KerasTensor`**: A symbolic placeholder representing the *future* output of a layer. It carries a `shape` but no actual data.
- **`Node`**: A record of one *call* to a layer. It links input `KerasTensor`s → output `KerasTensor`s and is stored on the layer's `_inbound_nodes` list.

When you write `outputs = Dense(32)(inputs)`, the layer's `__call__` method detects that `inputs` is a `KerasTensor`, builds the layer (if needed), computes the output shape symbolically, wraps it in a new `KerasTensor`, and records a `Node`. No NumPy computation occurs.

Later, `Model._init_graph` traverses the graph backward from the outputs to discover all reachable `Node`s and `Layer`s, producing a topological ordering used for forward and backward execution.

## Implementation Guide

### `KerasTensor` — `neutro/engine/node.py:3-13`

```python
class KerasTensor:
    def __init__(self, shape, node=None, name=None):
        self.shape = shape
        self.node = node      # The Node that produced this tensor
        self.name = name
```

- `shape` is a tuple like `(None, 32)` — the batch dimension is `None` (unknown until runtime).
- `node` is set when a `Node` is created and links back to the producing layer.

### `Node` — `neutro/engine/node.py:15-38`

```python
class Node:
    def __init__(self, layer, input_tensors, output_tensors):
        self.layer = layer
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        layer._inbound_nodes.append(self)
        # Link output tensors back to this node
        if isinstance(output_tensors, list):
            for t in output_tensors:
                t.node = self
        else:
            output_tensors.node = self
```

Key behaviors:
- **Registration**: The node registers itself on `layer._inbound_nodes`, enabling multi-parent graph traversal.
- **One layer, many nodes**: A shared layer used 3 times will have 3 entries in `_inbound_nodes`, each with different input/output tensors.
- **List outputs**: Layers like `Add` that take lists of inputs store the lists in `input_tensors`. Multi-output layers store lists in `output_tensors`.

### How `Layer.__call__` triggers Node creation — `neutro/layers/base.py:67-105`

The symbolic path (line 77-97):

```python
if is_symbolic:
    if not self.built:
        self.build(input_shapes)           # e.g., Dense.build((None, 10))
    output_shape = self.compute_output_shape(input_shapes)
    output_tensors = KerasTensor(shape=output_shape)
    Node(self, input_tensors=inputs, output_tensors=output_tensors)
    return output_tensors
```

This is a **zero-computation** path: no `forward` is called, only shape inference.

## Graph Discovery (`Model._init_graph`) — `neutro/models/base_model.py:25-62`

```python
def traverse(tensor):
    if hasattr(tensor, 'node') and tensor.node:
        node = tensor.node
        if node not in visited_nodes:
            visited_nodes.add(node)
            # Recursively visit inputs
            if isinstance(node.input_tensors, list):
                for t in node.input_tensors:
                    traverse(t)
            else:
                traverse(node.input_tensors)
            nodes_ordered.append(node)
```

This produces `_nodes_ordered` in **reverse topological order** (inputs before outputs). The backward pass iterates `reversed(_nodes_ordered)`.

## Usage Example

```python
from neutro.layers import Input, Dense
from neutro.models import Model
from neutro.engine.node import KerasTensor, Node

# Symbolic construction
inputs = Input(shape=(4,))          # returns a KerasTensor
x = Dense(8, activation='relu')(inputs)  # Layer.__call__ creates a Node
outputs = Dense(1)(x)

# Inspect the graph
print(type(inputs))          # <class 'KerasTensor'>
print(inputs.shape)          # (None, 4)
print(outputs.node.layer)    # Dense(1) — the final layer

# Model discovers nodes via traversal
model = Model(inputs=inputs, outputs=outputs)
print(len(model._nodes_ordered))  # Number of Nodes discovered
```

## References

- Chollet, F. (2015). **Keras** — the Functional API was introduced in Keras 1.0. [GitHub](https://github.com/keras-team/keras)
- Keras Functional API Guide. [Keras.io](https://keras.io/guides/functional_api/)
