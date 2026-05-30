# Activation Layer

## What does this layer do?

An Activation layer applies a non-linear function to its input. Without activation functions, a neural network would just be a series of linear transformations — no matter how many layers you stack, you could collapse them into a single matrix multiply. Activation functions introduce the non-linearity that gives neural networks their expressive power.

## The math

The Activation layer is a wrapper: it takes a string like `'relu'` and calls the corresponding function:

$$y = \phi(x)$$

Where:
- $x$ is the input (any shape)
- $\phi$ is an element-wise activation function (for most activations)
- $y$ has the same shape as $x$

### Backward pass (chain rule)

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \phi'(x)$$

For element-wise activations (ReLU, sigmoid, tanh), this is an element-wise product: each output gradient is multiplied by the derivative of the activation at the corresponding input position.

For **softmax**, the Jacobian is a full matrix (not diagonal): changing one input affects ALL outputs. So `gradient_fast` computes the full Jacobian-vector product.

## Walking through the code

### File: `neutro/layers/core/activation.py`

### Step 1: `__init__` — mapping strings to functions

```python
class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation = get_activation(activation)
```

🔍 **Line 7**: `get_activation('relu')` looks up the string in `neutro/activations/__init__.py` and returns an instance like `ReLU()`. This object has both a `__call__` (forward) and a `gradient` (backward) method.

You don't specify the input size here — like all `neutro` layers, shape is inferred when `forward` is first called.

### Step 2: no `build` needed

The Activation layer has **no learnable parameters**. It just applies a fixed function. So `build` is never overridden — it just marks `self.built = True`.

### Step 3: `forward` — applying the activation

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    return self.activation(inputs)
```

📐 **Shape**: Input shape = output shape. If inputs is `(B, D)`, output is `(B, D)`.

🔍 **Line 10**: `self.inputs = inputs` — We cache the input here because `backward` needs it to compute the gradient. For example, `ReLU.gradient(x)` returns `(x > 0).astype(float)`.

🔍 **Line 11**: `self.activation(inputs)` — This calls the activation's `__call__` method. For ReLU, it's `np.maximum(0, x)`. For softmax, it's the full softmax computation. Some activations (like softmax and sigmoid) also cache their output on `self` inside their `__call__`.

### Step 4: `backward` — two paths for gradient computation

```python
def backward(self, grad_output):
    if hasattr(self.activation, 'gradient_fast'):
        return self.activation.gradient_fast(self.inputs, grad_output)
    return grad_output * self.activation.gradient(self.inputs)
```

🔍 **Line 14**: `hasattr(self.activation, 'gradient_fast')` — Checks if this activation has a special fast gradient path. Currently only **softmax** has this.

**Path 1 — Element-wise activations (ReLU, sigmoid, tanh)**:

`grad_output * self.activation.gradient(self.inputs)`

📐 If `grad_output` is `(B, D)` and `gradient(self.inputs)` is `(B, D)`, then `(B, D) * (B, D)` → `(B, D)`. Broadcasting handles extra dimensions automatically.

- **ReLU**: `gradient(x) = (x > 0)`. Gradient of 1 for positive inputs, 0 for negative. "Dead ReLU" happens when many inputs are negative and the gradient is zero.
- **Sigmoid**: `gradient(x) = sigmoid(x) * (1 - sigmoid(x))`. Maximum gradient is 0.25 (at x=0), so deep sigmoid networks suffer from vanishing gradients.
- **Tanh**: `gradient(x) = 1 - tanh(x)^2`. Maximum gradient is 1.0 (at x=0), better than sigmoid.

**Path 2 — Softmax**:

`self.activation.gradient_fast(self.inputs, grad_output)`

Why is softmax special? For element-wise activations, output $y_i$ depends only on input $x_i$. But for softmax:

$$y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

The derivative is:

$$\frac{\partial y_i}{\partial x_j} = y_i (\delta_{ij} - y_j)$$

This is a full $C \times C$ Jacobian matrix per sample (where $C$ is the number of classes). `gradient_fast` computes the Jacobian-vector product efficiently:

```python
# For each sample:
s = out_flat[i].reshape(-1, 1)
jacobian = np.diagflat(s) - np.dot(s, s.T)    # (C, C)
res[i] = np.dot(grad_flat[i], jacobian)        # (C,) @ (C, C) -> (C,)
```

🧠 **Key insight**: `gradient_fast` exists because softmax is usually paired with cross-entropy loss. Their combined gradient is much simpler: just $(y - \text{target})$. But since the `Activation` layer doesn't know about the loss, it computes the full softmax Jacobian separately. In practice, you'd combine them for efficiency.

### Step 5: Convenience subclasses

```python
class ReLU(Activation):
    def __init__(self, **kwargs):
        super().__init__('relu', **kwargs)

class Softmax(Activation):
    def __init__(self, **kwargs):
        super().__init__('softmax', **kwargs)

class Sigmoid(Activation):
    def __init__(self, **kwargs):
        super().__init__('sigmoid', **kwargs)

class Tanh(Activation):
    def __init__(self, **kwargs):
        super().__init__('tanh', **kwargs)
```

These are just shortcuts so you can write `ReLU()` instead of `Activation('relu')`. They're functionally identical.

## Activation function reference

| Activation | Formula | Derivative | Gradient max | Notes |
|---|---|---|---|---|
| **ReLU** | $\max(0, x)$ | $1_{x > 0}$ | 1.0 | Can "die" (always zero) |
| **Sigmoid** | $1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ | 0.25 | Vanishing gradient |
| **Tanh** | $\tanh(x)$ | $1-\tanh^2(x)$ | 1.0 | Zero-centered |
| **Softmax** | $e^{x_i}/\sum e^{x_j}$ | Full Jacobian | — | For classification |

## Try it yourself

```python
from neutro.layers import Activation, ReLU, Sigmoid
import numpy as np

# Using the generic Activation layer
act = Activation('relu')
x = np.array([-2, -1, 0, 1, 2])
y = act(x)           # [0, 0, 0, 1, 2]
grad = np.array([1, 1, 1, 1, 1])
dx = act.backward(grad)  # [0, 0, 0, 1, 1] — gradient flows only for positive inputs

# Using convenience subclass
relu = ReLU()
y2 = relu(x)  # Same result
```

## What to read next

- `docs/layers/core/dense.md` — Dense layers are typically paired with activations
- `docs/activations/activations.md` — The activation function implementations themselves
- `docs/activations/softmax.md` — Deep dive into softmax and its Jacobian
