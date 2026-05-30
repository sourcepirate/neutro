# Layer Base Class

## Theory

Every neural network layer in `neutro` inherits from `neutro.layers.base.Layer`. The base class defines the **layer lifecycle**:

1. **Construction** (`__init__`): Set hyperparameters (units, kernel size, etc.). Do NOT allocate parameters yet.
2. **Build** (`build`): Allocate parameters based on the input shape (`self.params['W']`, `self.params['b']`, etc.).
3. **Call** (`__call__`): Dispatch — if inputs are symbolic `KerasTensor`s, do shape inference + node creation; if inputs are real NumPy arrays, run `forward`.
4. **Forward** (`forward`): Compute output from input.
5. **Backward** (`backward`): Compute gradient w.r.t. input and store gradients for parameters.

This deferred parameter allocation (build on first call) is the Keras convention: you don't need to specify input dimensions when constructing a layer — they are inferred from the data.

### Symbolic vs Eager Execution

A single `Layer.__call__` handles both modes:

- **Symbolic** (during model construction): Input is a `KerasTensor`. No NumPy computation happens; only shape inference and graph recording.
- **Eager** (during training/inference): Input is a NumPy array. The full forward pass runs.

## Implementation Guide

### File: `neutro/layers/base.py`

### `__init__` — line 4

```python
class Layer:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.trainable = True
        self.built = False
        self.params = {}      # {param_name: ndarray} — stores weights
        self.grads = {}       # {param_name: ndarray} — stores gradients
        self.input_shape = kwargs.get('input_shape')
        self.output_shape = None
        self._inbound_nodes = []  # Graph connectivity (Functional API)
```

- `built` starts as `False`. It becomes `True` after `build()` is called.
- `params` and `grads` are dicts so layers can have arbitrary parameter names (`W`, `b`, `gamma`, `beta`, etc.).

### `__call__` — line 67 — the dispatch hub

```python
def __call__(self, inputs, *args, **kwargs):
    from ..engine.node import KerasTensor, Node

    is_symbolic = isinstance(inputs, KerasTensor) or \
                  (isinstance(inputs, list) and any(isinstance(i, KerasTensor) for i in inputs))

    if is_symbolic:
        # Symbolic: build, infer shape, create Node
        if not self.built:
            self.build(input_shapes)
        output_shape = self.compute_output_shape(input_shapes)
        output_tensors = KerasTensor(shape=output_shape)
        Node(self, input_tensors=inputs, output_tensors=output_tensors)
        return output_tensors

    # Eager: build if needed, then forward
    if not self.built:
        self.build(inputs.shape if not isinstance(inputs, list) else [i.shape for i in inputs])
    return self.forward(inputs, *args, **kwargs)
```

Key detail: the symbolic path calls `build(input_shapes)` with tuples like `(None, 32)`. The eager path calls `build(inputs.shape)` with concrete shapes like `(64, 32)`.

### `sublayers` property — line 18

```python
@property
def sublayers(self):
    layers = []
    for attr_name in dir(self):
        attr = getattr(self, attr_name)
        if isinstance(attr, Layer):
            layers.append(attr)
        elif isinstance(attr, list):
            # Recurse into lists (e.g., TransformerBlock.ffn = [Dense, Dense])
            ...
    return layers
```

This is critical for:
- `count_params()`: sums params across all sublayers recursively.
- `_capture_layer_state()`: captures state of all sublayers for shared layer support.
- `_get_all_layers()`: collects every layer instance for the optimizer.

### `compute_output_shape` — line 55

Returns the expected output shape given an input shape. Used by:
- `Model.summary()` to build the layer table.
- Symbolic `__call__` to determine the output `KerasTensor.shape`.

### `count_params` — line 46

```python
def count_params(self):
    count = sum(p.size for p in self.params.values())
    for layer in self.sublayers:
        count += layer.count_params()
    return count
```

## Usage Example — Creating a Custom Layer

```python
from neutro.layers.base import Layer
import numpy as np

class MyDense(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.params['W'] = np.random.randn(input_shape[-1], self.units) * 0.01
        self.params['b'] = np.zeros(self.units)
        super().build(input_shape)  # sets self.built = True

    def forward(self, inputs):
        return np.dot(inputs, self.params['W']) + self.params['b']

    def backward(self, grad_output):
        self.grads['W'] = np.dot(self.inputs.T, grad_output)
        self.grads['b'] = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.params['W'].T)
```

## References

- Chollet, F. (2015). **Keras**: The Layer class API. [GitHub](https://github.com/keras-team/keras)
- Keras Custom Layers Guide. [Keras.io](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)
