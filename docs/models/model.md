# Sequential, Functional API, and Model

## Theory

`neutro` supports three model-building paradigms, mirroring Keras:

### 1. Sequential API

A linear stack of layers. Each layer's output is fed as input to the next. Simple and sufficient for many standard architectures (MLPs, CNNs, simple RNNs).

**Forward**: iterate `self.layers` in order, call each one.
**Backward**: iterate `self.layers` in reverse, call each `backward`.

### 2. Functional API

A graph (DAG) of layers where any layer output can feed into any other layer. Supports branching, merging, skip connections, multi-input, and multi-output models.

**Key concepts**:
- `KerasTensor`: a symbolic placeholder representing a future tensor.
- `Node`: records one call to a layer, linking inputs → output.
- `Model._init_graph`: traverses backward from outputs to discover all nodes and layers via topological sort.
- Forward: iterate nodes in topological order, compute each layer's output, store in `tensor_map`.
- Backward: iterate nodes in reverse topological order, accumulate gradients in `grad_map`.

### 3. Subclassed API

Override `Model` directly (write your own `forward` and `backward`). Used for architectures with custom control flow (e.g., UNet with skip connections, DiffusionModel).

**Inheritance**: `Model` inherits from `Layer`, so any `Model` instance can be used as a layer in another model (nested models).

## Implementation Guide

### File: `neutro/models/base_model.py`

### `Model.__init__`

```python
class Model(Layer):
    def __init__(self, inputs=None, outputs=None, name=None):
        super().__init__(name=name)
        self.layers = []
        self.optimizer = None
        self.loss_fn = None
        self.inputs = inputs
        self.outputs = outputs

        if inputs is not None and outputs is not None:
            self._init_graph(inputs, outputs)
```

- `Model` inherits from `Layer`, enabling nested models.
- If `inputs` and `outputs` are provided, this is a **Functional API** model and the graph is discovered immediately.

### Graph Discovery (`_init_graph`)

```python
def traverse(tensor):
    if hasattr(tensor, 'node') and tensor.node:
        node = tensor.node
        if node not in visited_nodes:
            visited_nodes.add(node)
            if isinstance(node.input_tensors, list):
                for t in node.input_tensors:
                    traverse(t)
            else:
                traverse(node.input_tensors)
            nodes_ordered.append(node)
```

This is a **post-order DFS** starting from the output tensors. The resulting `_nodes_ordered` is in **forward execution order** (inputs first). The backward pass iterates `reversed(_nodes_ordered)`.

Unique layers are collected from the nodes: `if node.layer not in self.layers`.

### Forward Pass

For Functional API models:

```python
tensor_map = {}
tensor_map[id(self.inputs)] = inputs  # Seed with user-provided data

for node in self._nodes_ordered:
    if isinstance(node.layer, InputLayer):
        continue  # InputLayer is a placeholder

    node_inputs = [tensor_map[id(t)] for t in node.input_tensors]
    output = node.layer.forward(node_inputs, training=training)

    # Capture state AFTER forward for shared layer support
    node.state = self._capture_layer_state(node.layer)

    tensor_map[id(node.output_tensors)] = output
```

**Key details**:
- `InputLayer` nodes are skipped; their values are placed in `tensor_map` before the loop.
- `node.state` is captured **after** `forward` runs, ensuring it stores the state from this specific call (not stale data from a previous call).
- The captured state uses `_capture_layer_state` which recurses into sublayers.

### Backward Pass

```python
grad_map = {}
grad_map[id(self.outputs)] = grad  # Seed with loss gradient

layer_grads_accumulator = {}  # For shared layers

for node in reversed(self._nodes_ordered):
    if isinstance(node.layer, InputLayer):
        continue

    node_grad_outputs = grad_map.get(id(node.output_tensors))

    # Restore state for this specific call (shared layer support)
    if hasattr(node, 'state'):
        self._restore_layer_state(node.layer, node.state)

    # Temporarily isolate this call's parameter gradients
    node.layer.grads = {}
    grad_inputs = node.layer.backward(node_grad_outputs)

    # Accumulate parameter gradients across shared calls
    l_id = id(node.layer)
    acc = layer_grads_accumulator.setdefault(l_id, {})
    for k, v in node.layer.grads.items():
        acc[k] = acc.get(k, 0) + v
    node.layer.grads = acc  # Point to accumulated dict

    # Propagate gradient to input nodes
    for i, t in enumerate(node.input_tensors):
        t_id = id(t)
        grad_map[t_id] = grad_map.get(t_id, 0) + grad_inputs[i]
```

**Key details**:
- **Gradient accumulation**: `layer_grads_accumulator` sums parameter gradients across multiple calls to the same shared layer.
- **State restoration**: Each node's captured state (from forward) is restored before its backward call, ensuring correct intermediate values (inputs, z, etc.).
- **Branching support**: If one tensor feeds into multiple downstream layers, `grad_map[t_id] += grad_inputs[i]` sums the gradients (the natural behavior for Add-branching).

### Shared Layer State Management

```python
@staticmethod
def _capture_layer_state(layer):
    state = {}
    stack = [layer]
    visited = set()
    while stack:
        l = stack.pop()
        if id(l) in visited: continue
        visited.add(id(l))
        sub = {k: v for k, v in l.__dict__.items()
               if k not in Model._STATE_EXCLUDE}
        state[id(l)] = sub
        for sl in l.sublayers:
            stack.append(sl)
    return state
```

This recursively captures the `__dict__` of every sublayer, keyed by `id()`. Excluded keys (`params`, `grads`, `built`, `input_shape`, etc.) are persistent architectural attributes that should not be restored.

### The `fit` Method

Supports three input modes:
1. **Single array**: `fit(x, y)` — standard training.
2. **List of arrays (MIMO)**: `fit([x1, x2], [y1, y2])` — multi-input, multi-output.
   - `is_mimo_x = isinstance(x, list)` detects list inputs.
   - For MIMO, batch slicing uses `[xi[start:end] for xi in x_shuffled]`.
3. **Generator**: `fit(generator)` — yields `(x_batch, y_batch)` tuples.

Loss is summed across multiple outputs (matching Keras behavior): `batch_loss = sum(self.loss_fn(y_batch[j], output[j])`.

### `evaluate`

Similarly handles MIMO: sums losses across outputs, falls back gracefully for metrics.

### `summary`

For Functional API models, displays a "Connected to" column showing each layer's upstream dependencies:

```text
Layer (type)         Output Shape    Param #   Connected to
Add (Add)           (None, 32)      0         input1, input2
```

### `_get_all_layers`

Returns all unique layer instances (deduplicated by `id()`) across the entire layer hierarchy, including sublayers. Used by the optimizer to update parameters.

### `Sequential`

```python
class Sequential(Model):
    def add(self, layer):
        if not self.layers:
            shape = layer.input_shape
            if len(shape) == 1: shape = (None,) + shape
            layer.build(shape)
        else:
            prev = self.layers[-1]
            input_shape = prev.compute_output_shape(prev.input_shape)
            layer.build(input_shape)
        self.layers.append(layer)
```

`Sequential` is a thin wrapper that builds layers in sequence based on the previous layer's output shape.

## Usage Examples

### Sequential

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```

### Functional — Skip Connection

```python
inputs = Input(shape=(32,))
x = Dense(32, activation='relu')(inputs)
skip = Dense(32)(x)
outputs = Add()([x, skip])
model = Model(inputs=inputs, outputs=outputs)
```

### Functional — MIMO

```python
i1 = Input(shape=(10,), name='input_a')
i2 = Input(shape=(10,), name='input_b')
merged = Add()([i1, i2])
o1 = Dense(1, name='out1')(merged)
o2 = Dense(2, name='out2')(merged)
model = Model(inputs=[i1, i2], outputs=[o1, o2])
model.fit([X1, X2], [Y1, Y2], epochs=10)
```

### Shared Layer

```python
shared = Dense(32, activation='relu')
inputs = Input(shape=(10,))
x1 = shared(inputs)
x2 = shared(x1)  # Same layer, second call
outputs = Dense(1)(x2)
model = Model(inputs=inputs, outputs=outputs)
```

### Nested Model

```python
inner = Model(inputs=inner_in, outputs=inner_out)
outer_input = Input(shape=(10,))
x = inner(outer_input)         # Model used as a layer
outputs = Dense(5)(x)
outer = Model(inputs=outer_input, outputs=outputs)
```

## References

- Chollet, F. (2015). **Keras**: Model and Sequential API. [GitHub](https://github.com/keras-team/keras)
- Keras Functional API Guide. [Keras.io](https://keras.io/guides/functional_api/)
