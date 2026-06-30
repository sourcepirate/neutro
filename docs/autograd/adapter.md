# `adapter` — Bridging Autograd to Existing Layers

## Why?

The existing `neutro` layers (`Dense`, `Conv2D`, etc.) store parameters in `self.params` dicts and gradients in `self.grads` dicts. Optimizers like `SGD` and `Adam` read from `layer.params` and `layer.grads` during `optimizer.step(layers)`.

The adapter lets you:
1. Create an autograd `Tensor` backed by a `Layer` parameter.
2. Run a forward/backward pass with autograd.
3. Copy the computed gradients back into the layer's `grads` dict.
4. Run the existing optimizer unchanged.

## Walkthrough

### `tensor_param(layer, param_name)` — Create a Tensor from a layer parameter

```python
def tensor_param(layer, param_name):
    param = layer.params[param_name]
    t = Tensor(param, requires_grad=True)
    return t
```

🔍 **Shares the underlying array**: `Tensor(param)` creates a Tensor whose `.data` is a *copy* of `param` (since `Tensor.__init__` calls `np.asarray(data, dtype=float)`, which copies). This means modifying `t.data` after creation does NOT modify `layer.params[param_name]` — you must call `update_param` to sync back.

### `update_param(layer, param_name, tensor)` — Copy data and grad back

```python
def update_param(layer, param_name, tensor):
    layer.params[param_name] = tensor.data.copy()
    if tensor.grad is not None:
        layer.grads[param_name] = tensor.grad.copy()
```

📐 **Shape matching**: `tensor.data` and `layer.params[param_name]` have the same shape by construction (since we initialized from it). The `.copy()` ensures the layer owns its data with no aliasing.

### `sync_grads(layer, param_tensors)` — Copy grads only

```python
def sync_grads(layer, param_tensors):
    for name, t in param_tensors.items():
        if t.requires_grad and t.grad is not None:
            layer.grads[name] = t.grad.copy()
```

🔍 **Gradient-only sync**: If you don't want to overwrite the parameter values (e.g., for gradient accumulation or checkpointing), use `sync_grads` instead of `update_param`.

### `autograd_step(optimizer, layers, param_tensor_dicts)` — Combined sync + step + zero

```python
def autograd_step(optimizer, layers, param_tensor_dicts=None):
    if param_tensor_dicts is None:
        param_tensor_dicts = {}
    for layer in layers:
        if id(layer) in param_tensor_dicts:
            sync_grads(layer, param_tensor_dicts[id(layer)])
    optimizer.step(layers)
    for layer in layers:
        if id(layer) in param_tensor_dicts:
            for t in param_tensor_dicts[id(layer)].values():
                t.zero_grad()
```

🔍 **Zero grads after step**: After the optimizer updates, we zero all autograd tensors' gradients to prepare for the next iteration.

## Usage Example

```python
from neutro.autograd import Tensor
from neutro.autograd.adapter import tensor_param, sync_grads
from neutro.layers import Dense
from neutro.optimizers import SGD

layer = Dense(1, activation=None)
layer.build((None, 2))
opt = SGD(learning_rate=1.0)

x = np.array([[1.0, 2.0]])
y = np.array([[3.0]])

for _ in range(50):
    W_t = tensor_param(layer, 'W')
    b_t = tensor_param(layer, 'b')
    
    pred = x @ W_t + b_t
    loss = ((pred - Tensor(y)) ** 2).sum()
    loss.backward()
    
    sync_grads(layer, {'W': W_t, 'b': b_t})
    opt.step([layer])
```

## Limitation

The adapter requires explicit per-parameter Tensor creation every forward pass. This is intentional — it keeps the layer's manual `backward` path untouched and makes the autograd usage explicit.

In a more integrated version, a custom `AutogradLayer` subclass of `Layer` could automatically wrap `self.params` as Tensors, but that would add complexity without educational value for v1.
