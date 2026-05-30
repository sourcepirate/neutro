# Input Layer and the `Input()` Function

## Theory

In the Functional API, every graph needs entry points — places where data enters the model. `Input()` creates a symbolic `KerasTensor` that acts as the root of the graph. The corresponding `InputLayer` is a no-op layer that simply passes data through; its role is purely structural.

`Input()` is a **convenience function** that:
1. Creates an `InputLayer` with the given shape.
2. Creates a `KerasTensor` as its symbolic output.
3. Records a `Node` connecting them.
4. Returns the `KerasTensor` for use in further layer calls.

The batch dimension is conventionally `None` (unknown until runtime), mirroring Keras behavior.

## Implementation Guide

### File: `neutro/layers/core/input_layer.py`

### `InputLayer` — line 4

```python
class InputLayer(Layer):
    def __init__(self, input_shape=None, name=None, **kwargs):
        super().__init__(name=name, input_shape=input_shape, **kwargs)
        if input_shape is not None:
            self.build(input_shape)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.built = True

    def forward(self, inputs, training=False):
        return inputs

    def backward(self, grad_output):
        return grad_output
```

- `forward` is the identity function — it returns its input unchanged.
- `backward` is also the identity — it passes the gradient straight through.
- `build` does not allocate any parameters; it only marks the layer as built.

### `Input()` function — line 28

```python
def Input(shape=None, name=None, **kwargs):
    if shape is None:
        raise ValueError("Please provide a shape for the Input.")

    if not isinstance(shape, tuple):
        shape = tuple(shape)

    # Keras style: prepend None for batch dimension if missing
    if len(shape) == 0 or shape[0] is not None:
        shape = (None,) + shape

    layer = InputLayer(input_shape=shape, name=name, **kwargs)
    output_tensor = KerasTensor(shape=shape, name=name)
    Node(layer, input_tensors=[], output_tensors=output_tensor)
    return output_tensor
```

Key behaviors:
- **Shape normalization**: If you pass `shape=(28, 28, 1)`, it becomes `(None, 28, 28, 1)`. This is the Keras convention: users specify the per-sample shape, and the batch dimension is prepended.
- **Empty input_tensors**: The `Node` created for `InputLayer` has an empty `input_tensors` list — it has no upstream layers.
- **The returned `KerasTensor`** has its `.node` set to this `Node`, so graph traversal can start from it.

### How InputLayer is handled during execution

In `Model.forward` (`neutro/models/base_model.py:217`):

```python
for node in self._nodes_ordered:
    if isinstance(node.layer, InputLayer):
        continue  # Skip — inputs are placed directly in tensor_map
```

InputLayer nodes are **skipped** during execution. Their values come from the model's input data, which is placed into `tensor_map` at the start of `forward`:

```python
tensor_map[id(self.inputs)] = inputs  # Placed before the loop
```

The same skip happens in `backward` (`line 314`): InputLayer nodes receive gradients but pass them back as the return value of the entire `backward` call.

## Usage Example

```python
from neutro.layers import Input, Dense, Add
from neutro.models import Model

# Single input
inputs = Input(shape=(28, 28, 1))    # KerasTensor of shape (None, 28, 28, 1)
x = Dense(32)(inputs)
model = Model(inputs=inputs, outputs=x)

# Multiple inputs
i1 = Input(shape=(10,), name='input_a')
i2 = Input(shape=(10,), name='input_b')
merged = Add()([i1, i2])
model = Model(inputs=[i1, i2], outputs=merged)
# forward expects [array_a, array_b]
```

## References

- Keras Functional API Guide: **Input()**. [Keras.io](https://keras.io/api/models/model/#functional-api)
