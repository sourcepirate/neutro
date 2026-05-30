# Input Layer and the `Input()` Function

## InputLayer — `neutro/layers/core/input_layer.py:4`

### What does this layer do?

`InputLayer` is the **entry point** of a computation graph. It is a no-op layer — it does not transform the data in any way. Its job is purely structural: it records a `Node` in the graph so the model knows this is where data enters.

### The math, in plain English

There is no math. InputLayer is the identity function:

$$
y = x
$$

Both forward and backward pass the data through unchanged.

### Walking through the code

#### `__init__`

```python
def __init__(self, input_shape=None, name=None, **kwargs):
    super().__init__(name=name, input_shape=input_shape, **kwargs)
    if input_shape is not None:
        self.build(input_shape)
```

🔍 **Line `super().__init__(name=name, ...)`**: Passes `name` and `input_shape` up to the base `Layer` class, where `self.input_shape` is stored.

🔍 **Line `if input_shape is not None: self.build(input_shape)`**: Unlike most layers (which wait for `build` to be called when data first flows through), `InputLayer` builds itself immediately because it already knows its shape.

#### `build`

```python
def build(self, input_shape):
    self.input_shape = input_shape
    self.built = True
```

🔍 **Line `self.input_shape = input_shape`**: Saves the shape. Note: `InputLayer` does **not** allocate any parameters (`self.params` stays empty). It's a parameter-free layer.

🔍 **Line `self.built = True`**: Marks the layer as built so subsequent calls don't trigger rebuild.

#### `forward`

```python
def forward(self, inputs, training=False):
    return inputs
```

🔍 **Identity function**: Input passes through completely unchanged. This is a **pass-through** layer — it exists only to connect the graph.

#### `backward`

```python
def backward(self, grad_output):
    return grad_output
```

🔍 **Identity again**: The gradient of the identity is 1, so $dL/dx = dL/dy$. The gradient passes through unchanged. This is the "last stop" for gradients during backpropagation — the model collects these gradients as the return value for the overall `backward` call.

---

## The `Input()` Function — `neutro/layers/core/input_layer.py:28`

### What does this function do?

`Input()` is a **convenience function** that you call at the top of the Functional API. It does four things in one shot:

1. Normalizes the shape (prepends `None` for the batch dimension).
2. Creates an `InputLayer` with that shape.
3. Creates a symbolic `KerasTensor` as the layer's output.
4. Records a `Node` connecting the layer to the tensor.
5. Returns the `KerasTensor` so you can feed it into other layers.

### Walking through the code

```python
def Input(shape=None, name=None, **kwargs):
    if shape is None:
        raise ValueError("Please provide a shape for the Input.")

    if not isinstance(shape, tuple):
        shape = tuple(shape)

    if len(shape) == 0 or shape[0] is not None:
        shape = (None,) + shape

    layer = InputLayer(input_shape=shape, name=name, **kwargs)

    output_tensor = KerasTensor(shape=shape, name=name)

    Node(layer, input_tensors=[], output_tensors=output_tensor)

    return output_tensor
```

🔍 **Line `if shape is None`**: The shape is required. Unlike some Keras variants, neutro does not infer the shape.

🔍 **Line `if not isinstance(shape, tuple): shape = tuple(shape)`**: If you pass a list like `[28, 28, 1]`, it's converted to a tuple `(28, 28, 1)`. This ensures consistent handling.

🔍 **Line `if len(shape) == 0 or shape[0] is not None: shape = (None,) + shape`**: This is the **batch dimension prepending** logic. In Keras convention, `Input(shape=(28, 28, 1))` means "each sample has shape `(28, 28, 1)`", and the batch dimension is implicitly `None` (unknown until runtime). So the stored shape becomes `(None, 28, 28, 1)`.

If you already pass `shape=(None, 28, 28, 1)`, the condition `shape[0] is not None` is False, so the shape is used as-is (no double wrapping).

🔍 **Line `layer = InputLayer(input_shape=shape, name=name, **kwargs)`**: Creates the actual layer instance. The `InputLayer.__init__` immediately calls `self.build(shape)`.

🔍 **Line `output_tensor = KerasTensor(shape=shape, name=name)`**: Creates a **symbolic tensor**. This is not real data — it's a placeholder that carries shape information. When you call other layers with this tensor (e.g., `Dense(32)(output_tensor)`), they use its shape to build themselves and record new `Node`s in the graph.

🔍 **Line `Node(layer, input_tensors=[], output_tensors=output_tensor)`**: Creates a graph node with **an empty `input_tensors` list**. This is the key difference from other nodes: InputLayer has no upstream layers — it's a **root** node. The `Node` constructor also sets `output_tensor.node = node`, linking the tensor back to this node.

📐 **Empty `input_tensors`**: When the model traverses the graph during execution, it starts from the model inputs (the `KerasTensor`s returned by `Input()`). The fact that InputLayer nodes have no input tensors signals "this is where execution begins."

🔍 **Line `return output_tensor`**: The function returns the **KerasTensor**, not the layer. This is what you assign to a variable:
```python
inputs = Input(shape=(28, 28, 1))  # inputs is a KerasTensor
x = Dense(32)(inputs)              # Dense receives the KerasTensor
```

### How InputLayer is handled during execution

InputLayer nodes are **skipped** during the model's forward and backward passes. Here's how:

```python
# In Model.forward (simplified):
tensor_map = {}
tensor_map[id(self.inputs)] = actual_data  # Place model inputs

for node in self._nodes_ordered:
    if isinstance(node.layer, InputLayer):
        continue  # SKIP — already in tensor_map
    # ... process other layers normally
```

🔍 **Line `tensor_map[id(self.inputs)] = actual_data`**: Before the execution loop, the model places the actual input data into the tensor map, keyed by the `KerasTensor`'s id. This is the starting point.

🔍 **Line `if isinstance(node.layer, InputLayer): continue`**: When the loop encounters an InputLayer node, it skips it. The tensor_map already has the data under the input KerasTensor's id, so no processing is needed.

The same skip happens in `backward`: InputLayer nodes receive (and pass through) gradients, but no gradient computation occurs within them.

### Why this design?

The `Input()` function + `InputLayer` design decouples **graph construction** from **data flow**:

- **Graph construction time** (when you call `Input()`): A `KerasTensor` is created. When you pass it to `Dense(32)`, the Dense layer creates a `Node` recording that connection. The graph is built symbolically — no real data moves.

- **Execution time** (when you call `model.fit()` or `model.predict()`): The model walks the graph, and for InputLayer nodes, it simply reads the actual data from the input list. The KerasTensor acts as a "key" to look up the real numpy array in the tensor_map.

## Usage Example

```python
from neutro.layers import Input, Dense, Add
from neutro.models import Model

# Single input
inputs = Input(shape=(28, 28, 1))    # Returns KerasTensor of shape (None, 28, 28, 1)
x = Dense(32)(inputs)
model = Model(inputs=inputs, outputs=x)

# Multiple inputs
i1 = Input(shape=(10,), name='input_a')
i2 = Input(shape=(10,), name='input_b')
merged = Add()([i1, i2])
model = Model(inputs=[i1, i2], outputs=merged)
# model.fit([x1_data, x2_data], y_data)
```

## References

- Keras Functional API Guide: **Input()**. [Keras.io](https://keras.io/api/models/model/#functional-api)
