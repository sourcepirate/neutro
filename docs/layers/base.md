# Layer Base Class

## What does this layer do?

Every neural network layer in `neutro` — whether it's a `Dense` layer, a `Conv2D` layer, or a `TransformerBlock` — inherits from `neutro.layers.base.Layer`. This base class defines the **layer lifecycle**: how a layer is constructed, how it creates its weights, how it processes data, and how nested layers inside it are discovered.

Think of it as the contract that every layer agrees to follow. If you want to write your own custom layer, you inherit from `Layer` and fill in four methods.

## The math, in plain English

There's no math for the base class itself — it's pure orchestration. But here is the **lifecycle** it enforces:

1. **`__init__`** — "Here are my settings" (e.g., "I want 64 units, ReLU activation")
2. **`build`** — "Now I know the input shape, so I'll create my weight matrices"
3. **`forward`** — "Give me real data, I'll compute the output"
4. **`backward`** — "Give me the gradient of the loss w.r.t. my output, I'll compute gradients for my weights and pass gradients back"

This separation lets you construct a layer *without* knowing the input dimensions upfront — the shape is inferred the first time you feed it data. This is the standard Keras convention.

## Walking through the code

### Step 1: `__init__` — setting the stage

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

🔍 **Line 4**: `self.name = name` — Just a label for debugging and `model.summary()`. If you don't give it one, that's fine; it defaults to `None`.

🔍 **Line 5**: `self.trainable = True` — Some layers (like a frozen embedding) shouldn't be updated during training. The optimizer checks this flag.

🔍 **Line 6**: `self.built = False` — This is a **gate**. It starts `False`, meaning "I haven't created my weights yet." After `build()` runs successfully, it flips to `True`. The check at line 100 uses this to decide whether to call `build()`.

🔍 **Lines 7-8**: `self.params = {}` and `self.grads = {}` — Dictionaries mapping string names to NumPy arrays. A `Dense` layer will store `params['W']` (the weight matrix) and `params['b']` (the bias vector). The gradients go into `grads['W']` and `grads['b']` after `backward` runs. Using dicts instead of fixed attributes means subclasses can have arbitrary parameter names (`gamma`, `beta`, `scale`, etc.).

🔍 **Line 9**: `self.input_shape = kwargs.get('input_shape')` — You *can* pass `input_shape` at construction time (like `Dense(64, input_shape=(128,))`), but usually it's inferred from the first call.

🔍 **Line 11**: `self._inbound_nodes = []` — Tracks graph connections for the Functional API. Every time you call a layer with a symbolic `KerasTensor`, a `Node` is created and appended here, recording which input tensors produced which output tensors.

### Step 2: `build` — creating learnable parameters

```python
def build(self, input_shape):
    self.input_shape = input_shape
    self.built = True
```

This is the **abstract stub** — subclasses override it. For example, `Dense.build` does:

```python
def build(self, input_shape):
    self.input_dim = input_shape[-1]
    self.params['W'] = self.kernel_initializer((self.input_dim, self.units))
    self.params['b'] = self.bias_initializer((self.units,))
    super().build(input_shape)  # <-- flips self.built = True
```

🔍 **Line 14**: `self.input_shape = input_shape` — Stores what shape of input this layer expects. This is used by `summary()` and by the symbolic call.

🔍 **Line 15**: `self.built = True` — Opens the gate. After this, `__call__` will skip `build()` on subsequent calls.

### Step 3: `__call__` — the dispatch hub

This is the most important method in the base class. It handles **two completely different modes** from a single entry point. The logic is factored into two helpers: `_symbolic_call` (graph-building) and `_eager_call` (computation).

```python
def __call__(self, inputs, *args, **kwargs):
    if self._is_symbolic_input(inputs):
        return self._symbolic_call(inputs)
    return self._eager_call(inputs, args, kwargs)
```

#### `_is_symbolic_input`

```python
def _is_symbolic_input(self, inputs):
    from ..engine.node import KerasTensor

    if isinstance(inputs, KerasTensor):
        return True
    return (isinstance(inputs, list)
            and any(isinstance(i, KerasTensor) for i in inputs))
```

🔍 **The fork**: If the input is a `KerasTensor` (or a list containing one), we're in "graph-building mode." If it's a real NumPy array, we're in "computation mode."

#### The symbolic branch — `_symbolic_call`

```python
def _symbolic_call(self, inputs):
    from ..engine.node import KerasTensor, Node

    if isinstance(inputs, list):
        input_shapes = [i.shape for i in inputs]
    else:
        input_shapes = inputs.shape

    if not self.built:
        self.build(input_shapes)

    output_shape = self.compute_output_shape(input_shapes)

    if isinstance(output_shape, list):
        output_tensors = [KerasTensor(shape=s) for s in output_shape]
    else:
        output_tensors = KerasTensor(shape=output_shape)

    Node(self, input_tensors=inputs, output_tensors=output_tensors)
    return output_tensors
```

When you use the Functional API like:

```python
inputs = Input(shape=(128,))
x = Dense(64)(inputs)
```

The `KerasTensor` called `inputs` is passed to `Dense.__call__`. No actual numbers flow through — just shape information.

🔍 **`input_shapes = ...`** — Extracts the shape from the symbolic tensor. Shapes look like `(None, 128)` where `None` means "unknown batch size."

🔍 **`self.build(input_shapes)`** — Allocates weight matrices with the correct dimensions, but the actual *values* don't matter here. What matters is that `self.params['W']` now exists with the right shape.

🔍 **`self.compute_output_shape(input_shapes)`** — Asks the layer: "If I give you input shape `(None, 128)`, what will my output shape be?" For a `Dense(64)` layer, the answer is `(None, 64)`.

🔍 **Creating output `KerasTensor`s** — Wraps the computed output shape into a new symbolic tensor. This tensor will be passed as input to the *next* layer.

🔍 **`Node(self, input_tensors=inputs, output_tensors=output_tensors)`** — Records the connection in the computation graph. This `Node` links "the input tensor(s)" to "the output tensor(s)" through "this layer." Later, `Model` walks these nodes to figure out the topology — which layers connect to which, what the forward pass order should be, and what the inputs/outputs of the whole model are.

#### The eager branch — `_eager_call`

```python
def _eager_call(self, inputs, args, kwargs):
    from neutro.autograd import as_tensor as _as_tensor

    t_inputs = _as_tensor(inputs)
    if not self.built:
        if isinstance(t_inputs, list):
            self.build([i.shape for i in t_inputs])
        else:
            self.build(t_inputs.shape)
    self._last_inputs = t_inputs
    self._last_args = args
    self._last_kwargs = kwargs
    output = self.forward(t_inputs, *args, **kwargs)
    self._last_output = output
    return output
```

When you call a layer directly with real data:

```python
x = np.random.randn(32, 128)
y = layer(x)  # forwards! actual computation!
```

🔍 **`if not self.built: self.build(t_inputs.shape)`** — First call? Build the weights using the actual concrete shape (e.g., `(32, 128)`). Note that `inputs.shape` here is a real tuple of integers, not a symbolic shape with `None`.

🔍 **`self._last_inputs = t_inputs`** — Caches the input for the `backward` method, which re-runs the forward pass inside a `GradientTape` to compute gradients.

🔍 **`return self.forward(t_inputs, *args, **kwargs)`** — Delegates to the subclass's actual computation. This is where the matrix multiply happens, where the convolution runs, where the attention scores are computed.

### Step 4: `sublayers` — finding nested layers

```python
@property
def sublayers(self):
    """Return all nested layers contained within this layer.

    Traverses instance attributes, including sublayers stored in lists,
    tuples, or dictionaries, with cycle protection and de-duplication.
    """
    layers = []
    visited_ids = {id(self)}
    stack = []
    for attr in vars(self).values():
        stack.append(attr)

    while stack:
        attr = stack.pop()
        if isinstance(attr, Layer):
            if id(attr) not in visited_ids:
                visited_ids.add(id(attr))
                layers.append(attr)
                for nested in vars(attr).values():
                    stack.append(nested)
        elif isinstance(attr, (list, tuple, set)):
            stack.extend(attr)
        elif isinstance(attr, dict):
            stack.extend(attr.values())
    return layers
```

This property is how `neutro` discovers layers inside layers. Consider a `TransformerBlock`:

```python
class TransformerBlock(Layer):
    def __init__(self, ...):
        self.attention = MultiHeadAttention(...)
        self.ffn = [Dense(512), Dense(512, activation='relu')]
```

When the optimizer needs to find **all** trainable parameters, it calls `sublayers` on the top-level model. The property:

1. Collects every instance attribute of the layer (via `vars(self)`) into a work stack.
2. If an attribute is a `Layer` instance, it collects it and pushes that layer's own attributes onto the stack — this catches `self.attention`, `self.norm`, etc. The `visited_ids` set de-duplicates, so a shared layer used in multiple branches is only reported once.
3. If an attribute is a **list, tuple, or set**, it pushes each element — this catches `self.ffn = [Dense(512), Dense(512)]` and the lists-of-lists used by `MoELayer` (a list of expert lists).
4. If an attribute is a **dict**, it pushes the values.

🔍 **Why is this important?** Without `sublayers`, a `TransformerBlock` would report only its own `params` dict (which is empty — it delegates everything to sublayers). With `sublayers`, the optimizer can traverse the full hierarchy and find every weight matrix in every attention head and every feed-forward layer. The cycle protection (`visited_ids`) and de-duplication prevent infinite recursion and double-counting when layers are shared.

### Step 5: `count_params` — the recursive parameter counter

```python
def count_params(self):
    count = sum(p.size for p in self.params.values())
    for layer in self.sublayers:
        count += layer.count_params()
    return count
```

🔍 **Line 50**: `sum(p.size for p in self.params.values())` — Counts the parameters owned directly by this layer. For a `Dense(64, input_dim=128)`, that's `128 * 64 + 64 = 8256` (weights + biases).

🔍 **Lines 51-52**: `for layer in self.sublayers: count += layer.count_params()` — Recursively counts parameters in all sublayers. A `TransformerBlock` calls `count_params` on each attention head, each feed-forward layer, and each normalization layer. Those sublayers might have their *own* sublayers (like `LayerNormalization` which has `gamma` and `beta`), so the recursion keeps going.

This gives you the total parameter count you see in `model.summary()`.

### Step 6: `compute_output_shape` and `backward`

```python
def compute_output_shape(self, input_shape):
    if hasattr(self, 'output_shape') and self.output_shape is not None:
        return self.output_shape
    return input_shape
```

🔍 **Default behavior** — if no `output_shape` was explicitly set, assume the output shape equals the input shape. Subclasses like `Dense` override this to return `(*input_shape[:-1], units)`.

#### `backward`

The base `Layer.backward` provides a **generic gradient mechanism** for layers whose `forward` is written with autograd `Tensor` operations. It re-runs the forward pass inside a `GradientTape` to recover the parameter gradients via reverse-mode differentiation:

```python
def backward(self, grad_output):
    from neutro.autograd import GradientTape, Tensor as AT

    t_inputs = self._captured_inputs()          # autograd view of the last inputs
    if t_inputs is None:
        return np.asarray(grad_output)

    fwd_args = getattr(self, '_last_args', ())
    fwd_kwargs = getattr(self, '_last_kwargs', {}).copy()
    fwd_kwargs.pop('kv_cache', None)            # runtime bookkeeping, not differentiable
    fwd_kwargs.pop('layer_id', None)

    all_sources = list(self._collect_tensor_params().values())
    # ... include t_inputs in all_sources ...

    with GradientTape() as tape:
        for source in all_sources:
            tape.watch(source)
        output = self.forward(t_inputs, *fwd_args, **fwd_kwargs)
        g_t = AT(np.asarray(grad_output))
        loss = (output * g_t).sum()             # dot product = weighted gradient probe

    tape.gradient(loss, all_sources)
    # ... copy param_value.grad into layer.grads for every sublayer ...
    return t_inputs.grad.copy()                 # gradient w.r.t. the inputs
```

🔍 **Why re-run forward?** The gradient of the weighted probe `loss = (output * g_t).sum()` w.r.t. any parameter is exactly the upstream gradient `g_t` propagated through the layer's computation graph. This gives a correct, dependency-free backward pass for any forward written purely with autograd ops.

🔍 **`_captured_inputs` / `_collect_tensor_params`** — helper methods that rebuild the input `Tensor`s and gather every autograd parameter across the layer tree (including sublayers). This is what makes `backward` work for nested layers like `TransformerBlock` without each sublayer reimplementing graph traversal.

## Putting it all together

Here's what happens when you write:

```python
layer = Dense(64, activation='relu')
x = np.random.randn(32, 128)
y = layer(x)
```

1. `Layer.__init__` runs (via `super().__init__()` inside `Dense.__init__`). `built = False`, `params = {}`, `grads = {}`.
2. `Dense.__init__` stores `self.units = 64` and creates the activation function object.
3. `layer(x)` invokes `Layer.__call__`.
4. `__call__` checks: is `x` a `KerasTensor`? No, it's a NumPy array → **eager branch**.
5. Is `self.built` `False`? Yes → calls `self.build((32, 128))`.
6. `Dense.build` allocates `params['W']` with shape `(128, 64)` and `params['b']` with shape `(64,)`, then calls `super().build()` which sets `self.built = True`.
7. `__call__` calls `self.forward(x)`.
8. `Dense.forward` computes `np.dot(x, W) + b`, applies ReLU, caches `self.inputs` and `self.z`, returns the output.
9. Later, `layer.backward(grad_output)` uses those cached values to compute weight gradients.

## 🚫 No Autograd — You Write the Gradients

This is the single most important thing to understand about `neutro`:

**There is no automatic differentiation engine.**

In PyTorch, you write:

```python
y = x @ W + b      # PyTorch traces this into a graph
y.backward()       # PyTorch automatically computes gradients for W and b
```

In `neutro`, you write both `forward` AND `backward`:

```python
def forward(self, x):
    self.inputs = x
    return x @ self.params['W'] + self.params['b']

def backward(self, grad_output):
    self.grads['W'] = self.inputs.T @ grad_output
    self.grads['b'] = np.sum(grad_output, axis=0)
    return grad_output @ self.params['W'].T
```

Why? Because every matrix multiplication you write in `backward` — every `@`, every `np.sum`, every `reshape` — is an explicit application of the **chain rule**. You are not calling `loss.backward()`. You *are* the autograd engine.

This means:
- **If you add a new layer**, you must implement `backward` yourself — no framework will do it for you.
- **If you change the forward pass**, you must update backward to match. Every new line in `forward` probably needs a corresponding line in `backward`.
- **If backward gives wrong shapes**, you'll get a NumPy shape mismatch error — not a cryptic autograd graph error. You'll learn to think in shapes.
- **Every value you cache on `self` in `forward`** (like `self.inputs` or `self.z`) is cached for one reason: `backward` needs it. There is no tape, no graph, no magic — just stored NumPy arrays and chain rule math.

This is the defining educational feature of the library. You can't hand-wave through gradient descent here. You must understand where gradients come from.

## Try it yourself

Here's how you'd create a custom `MyDense` layer from scratch:

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
        self.inputs = inputs  # cached for backward
        return np.dot(inputs, self.params['W']) + self.params['b']

    def backward(self, grad_output):
        self.grads['W'] = np.dot(self.inputs.T, grad_output)
        self.grads['b'] = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.params['W'].T)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

# Try it
layer = MyDense(units=32)
x = np.random.randn(16, 64)
y = layer(x)                   # forward: (16, 64) -> (16, 32)
print(y.shape)                 # (16, 32)
print(layer.count_params())    # 64*32 + 32 = 2080
```

Notice that we:
1. Called `super().__init__(**kwargs)` in `__init__` so the base class sets up `self.built`, `self.params`, etc.
2. Called `super().build(input_shape)` at the end of `build` to flip the `built` flag.
3. Stored `self.inputs` in `forward` because `backward` needs it.
4. Implemented all four lifecycle methods.

## What to read next

- **`neutro/layers/core/dense.md`** — See a concrete example: how `Dense` implements this lifecycle with a full forward/backward pass, including how activations chain into the gradient computation.
- **`neutro/layers/core/dropout.md`** — A different kind of layer: stochastic (random) during training, deterministic during inference.
- **`neutro/models/base_model.md`** — How `Model` uses `sublayers` and `count_params` to orchestrate training loops.
