# Dense Layer

## Theory

A Dense (fully-connected) layer computes a linear transformation followed by an optional activation:

$$y = \phi(xW + b)$$

Where:
- $x \in \mathbb{R}^{B \times D}$ is the input (batch $B$, input dimension $D$)
- $W \in \mathbb{R}^{D \times U}$ is the weight matrix (learned)
- $b \in \mathbb{R}^{U}$ is the bias vector (learned)
- $\phi$ is an element-wise activation function (ReLU, sigmoid, tanh, or none)
- $y \in \mathbb{R}^{B \times U}$ is the output

### Backward Pass

The gradients are:

$$\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial y}$$

$$\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial y}$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T$$

If an activation function $\phi$ is present, the gradient is first passed through $\phi'$ before these equations.

## Implementation Guide

### File: `neutro/layers/core/dense.py`

### `__init__` — line 7

```python
class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
```

- `units`: number of output neurons.
- `activation`: a string like `'relu'` → mapped to an activation function via `get_activation()`.
- Weight initialization is deferred to `build()`.

### `build` — line 15

```python
def build(self, input_shape):
    self.input_dim = input_shape[-1]
    self.params['W'] = self.kernel_initializer((self.input_dim, self.units))
    if self.use_bias:
        self.params['b'] = self.bias_initializer((self.units,))
    super().build(input_shape)
```

Parameters are allocated here, not in `__init__`. This is the standard Keras pattern: the shape is inferred from the first call.

### `forward` — line 26

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    self.z = np.dot(inputs, self.params['W'])
    if self.use_bias:
        self.z += self.params['b']
    if self.activation:
        return self.activation(self.z)
    return self.z
```

- `self.inputs` is cached for use in `backward`.
- `self.z` is cached for use in activation backpropagation.
- The activation function (`self.activation`) is called as a callable; it may be a `Layer` instance with its own forward/backward.

### `backward` — line 36

```python
def backward(self, grad_output):
    if self.activation:
        grad_output = grad_output * self.activation.gradient(self.z)

    inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])
    grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])

    self.grads['W'] = np.dot(inputs_flat.T, grad_output_flat)
    if self.use_bias:
        self.grads['b'] = np.sum(grad_output_flat, axis=0)

    return np.dot(grad_output, self.params['W'].T)
```

- For activation backprop, the Jacobian of the activation is element-wise multiplied with `grad_output` (most activations like ReLU, sigmoid, tanh are element-wise; Softmax is handled separately via `gradient_fast`).
- The matrix multiplications are the exact implementation of the gradient equations above.
- The return value is the gradient with respect to the input, which is passed to the previous layer.

## Usage Example

```python
from neutro.layers import Dense
import numpy as np

layer = Dense(units=64, activation='relu')
x = np.random.randn(32, 128)  # (batch, input_dim)
y = layer(x)                   # forward, shape (32, 64)
grad = np.random.randn(32, 64)
dx = layer.backward(grad)      # gradient w.r.t. x, shape (32, 128)
```

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). **Deep Learning**. Chapter 6: Deep Feedforward Networks. [Deep Learning Book](https://www.deeplearningbook.org/)
