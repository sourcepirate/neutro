# Dense Layer

## What does this layer do?

A Dense (or "fully-connected") layer connects every input neuron to every output neuron. You give it a vector (or a batch of vectors), it multiplies by a weight matrix, adds a bias, and optionally runs an activation function like ReLU or sigmoid.

Think of it as the "basic building block" of neural networks — most models start and end with one or more Dense layers.

## The math, in plain English

$$y = \phi(x W + b)$$

Let's unpack every symbol:

- **$x$** — Your input. Shape `(batch_size, input_dim)`. Each row is one data point (e.g., a 128-dimensional feature vector for one image).
- **$W$** — The weight matrix. Shape `(input_dim, units)`. Every entry $W_{ij}$ controls how much input neuron $i$ contributes to output neuron $j$. These are **learned** during training.
- **$b$** — The bias vector. Shape `(units,)`. An offset added to each output neuron. Also **learned**.
- **$xW$** — Matrix multiply: `(batch, input_dim) @ (input_dim, units)` → `(batch, units)`. This is a **linear transformation** — it rotates and scales the input space.
- **$xW + b$** — The bias is *broadcast* across the batch (added to every row). This gives each output neuron a baseline firing threshold.
- **$\phi$** — An activation function applied element-wise (to each number independently). ReLU turns negatives to zero, sigmoid squashes values between 0 and 1, etc. This is where **non-linearity** comes from — without it, stacking Dense layers would be the same as one big linear transformation.
- **$y$** — Your output. Shape `(batch, units)`. Each row is a `units`-dimensional transformed representation of the input.

### How gradients flow backward

During training, we need to adjust $W$ and $b$ to reduce the loss. The gradient formulas are:

$$\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial y}$$

$$\frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial y}$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W^T$$

Here $\frac{\partial L}{\partial y}$ is the gradient coming *from the next layer* (the "upstream gradient"). The three formulas tell us:

1. **Weight gradient**: Transpose the input and multiply by the upstream gradient. Shape: `(input_dim, batch) @ (batch, units)` → `(input_dim, units)` — exactly the same shape as $W$.
2. **Bias gradient**: Sum the upstream gradient over the batch dimension. The bias is added to every sample, so its gradient is the sum of all per-sample gradients.
3. **Input gradient**: Multiply the upstream gradient by $W^T$. This gets passed to the previous layer so it can compute *its* weight gradients.

If an activation $\phi$ is present, the chain rule says we must first multiply the upstream gradient by $\phi'(z)$ (the derivative of the activation at $z = xW + b$):

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \odot \phi'(z)$$

where $\odot$ is element-wise multiplication, and then use $\frac{\partial L}{\partial z}$ in place of $\frac{\partial L}{\partial y}$ in the three formulas above.

## Walking through the code

### Step 1: `__init__` — setting the stage

```python
class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
```

🔍 **Line 7**: `super().__init__(**kwargs)` — Calls `Layer.__init__`, which sets `self.built = False`, creates empty `self.params = {}` and `self.grads = {}`. The `**kwargs` lets you pass `name='my_dense'` or `input_shape=(128,)` which the base class knows how to handle.

🔍 **Line 8**: `self.units = units` — We store this now because we won't know the weight shapes until `build()` runs (since we don't know `input_dim` yet). But `units` is a hyperparameter *we* choose, so it goes in `__init__`.

🔍 **Line 9**: `self.activation = get_activation(activation)` — `activation` is a string like `'relu'`, `'sigmoid'`, or `None`. The `get_activation` function looks it up and returns an `Activation` object (e.g., a `ReLU` instance). This object has two key methods: `__call__` (the forward pass) and `gradient` (the derivative for backprop). If `activation=None`, `get_activation` returns `None` and we skip the activation step.

🔍 **Line 10**: `self.use_bias = use_bias` — Some layers (like the layer right before a softmax) don't need a bias. Storing this flag lets `build` decide whether to allocate `params['b']`.

🔍 **Lines 11-12**: `self.kernel_initializer` and `self.bias_initializer` — These are *strategy objects* that know how to create weight matrices with sensible starting values. `glorot_uniform` draws from a uniform distribution scaled by the number of input/output neurons. The actual initialization is deferred to `build`.

### Step 2: `build` — creating the learnable parameters

```python
def build(self, input_shape):
    self.input_dim = input_shape[-1]
    self.params['W'] = self.kernel_initializer((self.input_dim, self.units))
    if self.use_bias:
        self.params['b'] = self.bias_initializer((self.units,))
    super().build(input_shape)
```

🔍 **Line 16**: `self.input_dim = input_shape[-1]` — We grab the last dimension of the input shape. If input is `(32, 128)` (batch of 32, each 128-dimensional), then `input_dim = 128`. But what if the input is 3D, like `(32, 10, 64)`? Then `input_dim = 64` — we only care about the **last** dimension because Dense operates on the *last axis*.

🔍 **Line 17**: `self.params['W'] = self.kernel_initializer((self.input_dim, self.units))` — Here's where the weight matrix is actually created. Shape is `(input_dim, units)`:

```
          units (= 64)
    ┌─────────────────┐
    │ W[0,0]  W[0,1]  │
D   │ W[1,0]  W[1,1]  │   Each column j: "how to compute
i   │   ...     ...    │   output neuron j from all inputs"
m   │                  │
(=128)│ W[127,0] W[127,63]│
    └─────────────────┘
```

Think of each **column** of $W$ as a set of weights that produce one output neuron. The input `input_dim` must match the last dimension of whatever data comes in.

🔍 **Line 19**: `self.params['b'] = self.bias_initializer((self.units,))` — The bias is a 1D vector of length `units`. When added to `xW` (shape `(batch, units)`), NumPy broadcasts it across the batch dimension automatically.

🔍 **Line 20**: `super().build(input_shape)` — This sets `self.built = True`. After this line, the layer won't call `build` again. It also stores `self.input_shape` for `summary()`.

### Step 3: `forward` — the main computation

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

🔍 **Line 27**: `self.inputs = inputs` — We **cache** the input here. Why? Because `backward` (line 36) needs it to compute `self.grads['W'] = np.dot(inputs_flat.T, grad_output_flat)`. The input isn't available during `backward` unless we saved it now.

🔍 **Line 28**: `self.z = np.dot(inputs, self.params['W'])` — The core computation. Let's trace the shapes:

📐 **Shape walkthrough**: `inputs` is `(B, D)` where `B = batch_size, D = input_dim`. `self.params['W']` is `(D, U)` where `U = units`. `np.dot((B, D), (D, U))` → `(B, U)`. Each output row is the input row multiplied by the weight matrix.

But wait — `inputs` might be 3D! For example, a `(B, T, D)` sequence where `T` is sequence length. That's fine: `np.dot` treats the first dimensions as batch dimensions, so `(B, T, D) @ (D, U)` → `(B, T, U)`. The same weight matrix is applied at every position in the sequence.

🔍 **Lines 29-30**: `if self.use_bias: self.z += self.params['b']` — Adding the bias vector. NumPy broadcasting means `(B, U) += (U,)` adds the same bias to every row. If `inputs` was 3D, this broadcasts as `(B, T, U) += (U,)`.

🔍 **Line 32**: `self.z` is cached for a *different* reason than `self.inputs`. It stores the **pre-activation** values ($xW + b$, before the activation function). In `backward`, line 41, we compute `self.activation.gradient(self.z)` — the derivative of the activation evaluated at these pre-activation values. Without caching `self.z`, we'd need to recompute it in backward.

🔍 **Line 33**: `return self.activation(self.z)` — Calls the activation function's `__call__`, which applies it element-wise. If `activation` is `None`, we skip to line 34 and return `self.z` directly.

### Step 4: `backward` — learning from mistakes

```python
def backward(self, grad_output):
    if self.activation:
        if hasattr(self.activation, 'gradient_fast'):
            grad_output = self.activation.gradient_fast(self.z, grad_output)
        else:
            grad_output = grad_output * self.activation.gradient(self.z)

    inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])
    grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])

    self.grads['W'] = np.dot(inputs_flat.T, grad_output_flat)
    if self.use_bias:
        self.grads['b'] = np.sum(grad_output_flat, axis=0)

    return np.dot(grad_output, self.params['W'].T)
```

Let's break this down piece by piece.

#### Step 4a: Handle the activation gradient

```python
    if self.activation:
        if hasattr(self.activation, 'gradient_fast'):
            grad_output = self.activation.gradient_fast(self.z, grad_output)
        else:
            grad_output = grad_output * self.activation.gradient(self.z)
```

🔍 **Lines 37-41**: The **chain rule**. We have the upstream gradient `grad_output` (shape `(B, U)`), which is $\frac{\partial L}{\partial y}$ — the gradient of the loss w.r.t. the *activated* output.

But the weight gradient formulas use $\frac{\partial L}{\partial z}$ — the gradient w.r.t. the *pre-activation* values. The chain rule says:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} = \frac{\partial L}{\partial y} \odot \phi'(z)$$

For element-wise activations (ReLU, sigmoid, tanh), $\phi'(z)$ is just the derivative evaluated at each element, and the multiplication is element-wise.

🔍 **Line 41**: `grad_output * self.activation.gradient(self.z)` — This is the standard path. `self.activation.gradient(self.z)` returns an array of the same shape as `self.z` (cached in forward at line 28), containing $\phi'(z)$ at each element. For ReLU, this is `(z > 0)` — a mask of 1s and 0s. Element-wise multiply with `grad_output` zeros out gradients for ReLU'd neurons that were originally negative.

🔍 **Lines 38-39**: `self.activation.gradient_fast(self.z, grad_output)` — A special path for **Softmax**. Why? Because Softmax's Jacobian isn't element-wise — each output depends on *all* inputs, so the full Jacobian is a `(U, U)` matrix per sample. The `gradient_fast` method on `Softmax` computes the matrix-vector product `grad_output @ J_softmax` efficiently without constructing the full `(U, U)` Jacobian explicitly (well, in this educational implementation it does construct it, but you can imagine a more efficient version). The standard `gradient` method (which returns `s * (1 - s)`) would give the wrong answer for softmax — it's only correct for element-wise sigmoid.

#### Step 4b: Flatten for multi-dimensional inputs

```python
    inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])
    grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
```

🔍 **Lines 43-44**: Why the reshape? Let's say the input was 3D: `inputs.shape = (8, 10, 64)` — batch of 8 sequences, each 10 tokens, each token 64-dimensional. The forward pass produced `z.shape = (8, 10, 32)` and `grad_output.shape = (8, 10, 32)`.

To compute `self.grads['W']`, we need `inputs.T @ grad_output`. But `inputs` is 3D and `grad_output` is 3D — we can't just transpose and multiply.

So we **flatten the batch dimensions**:

📐 **Shape walkthrough**: `inputs` `(8, 10, 64)` → `.reshape(-1, 64)` → `(80, 64)`. The `-1` tells NumPy: "figure out the size automatically" — `8 * 10 = 80`. Now `grad_output` `(8, 10, 32)` → `.reshape(-1, 32)` → `(80, 32)`.

Now the matrix multiply works: `(64, 80) @ (80, 32)` → `(64, 32)` = `(input_dim, units)`, which is exactly the shape of `self.params['W']`.

The bias gradient also benefits: `np.sum(grad_output_flat, axis=0)` sums over all 80 positions, giving shape `(32,)` = `(units,)`.

#### Step 4c: Compute the weight gradient

```python
    self.grads['W'] = np.dot(inputs_flat.T, grad_output_flat)
```

🔍 **Line 46**: This implements $\frac{\partial L}{\partial W} = x^T \cdot \frac{\partial L}{\partial z}$.

📐 **Shape walkthrough**: `inputs_flat.T` is `(D, B')` where `B' = B * T` (all positions flattened). `grad_output_flat` is `(B', U)`. `np.dot((D, B'), (B', U))` → `(D, U)` — matching the shape of `W`.

🔍 **Why it works**: Each element `(i, j)` of the result is `sum over batch of inputs_flat[k, i] * grad_output_flat[k, j]`. This is exactly the average (well, sum) co-variation of input feature `i` and output error `j` — if they tend to be positive together, the weight should increase.

#### Step 4d: Compute the bias gradient

```python
    if self.use_bias:
        self.grads['b'] = np.sum(grad_output_flat, axis=0)
```

🔍 **Line 48**: Summing over `axis=0` (the batch/time dimension). For each output neuron `j`, `self.grads['b'][j]` is the sum of `grad_output_flat[:, j]` over all samples. This implements $\frac{\partial L}{\partial b_j} = \sum_{\text{batch}} \frac{\partial L}{\partial z_j}$.

#### Step 4e: Compute the input gradient (for the previous layer)

```python
    return np.dot(grad_output, self.params['W'].T)
```

🔍 **Line 50**: This implements $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot W^T$. We use the **original** (non-flattened) `grad_output` here because the previous layer expects the same number of dimensions as its output had.

📐 **Shape walkthrough**: If input was 3D `(B, T, D)`, then `grad_output` is `(B, T, U)` and `W.T` is `(U, D)`. `np.dot((B, T, U), (U, D))` → `(B, T, D)` — matching the original input shape perfectly.

If input was 2D `(B, D)`: `np.dot((B, U), (U, D))` → `(B, D)` — also correct.

## Putting it all together

Here's the full lifecycle when you call `layer(x)` on a Dense layer with ReLU activation:

1. **`Layer.__call__`** is invoked with your input data.
2. It checks `self.built` — on the first call, it's `False`, so `build(inputs.shape)` runs, creating `params['W']` and `params['b']`.
3. It calls `self.forward(inputs)`, which is `Dense.forward`.
4. **`Dense.forward`**:
   - Caches `self.inputs = inputs` (needed later in backward line 43)
   - Computes `self.z = np.dot(inputs, W)` → the pre-activation values
   - Adds bias: `self.z += b`
   - Applies ReLU: `return np.maximum(0, self.z)`
5. Later, `layer.backward(grad_output)` is called:
   - Multiplies `grad_output` by ReLU's gradient `(z > 0)` — zeroing out gradients for negative pre-activations
   - Flattens `self.inputs` and `grad_output` to handle any number of dimensions
   - Computes `grads['W'] = inputs_flat.T @ grad_output_flat`
   - Computes `grads['b'] = sum(grad_output_flat, axis=0)`
   - Returns `grad_output @ W.T` — the gradient for the previous layer

The optimizer then uses `self.grads['W']` and `self.grads['b']` to update `self.params['W']` and `self.params['b']`.

## Try it yourself

```python
from neutro.layers import Dense
import numpy as np

# Create a Dense layer with 64 output units and ReLU activation
layer = Dense(units=64, activation='relu')

# Generate random input: batch of 32, each 128-dimensional
x = np.random.randn(32, 128)

# Forward pass — this triggers build on first call
y = layer(x)
print(f"Output shape: {y.shape}")                # (32, 64)
print(f"Parameters: {layer.count_params()}")      # 128*64 + 64 = 8256

# Simulated upstream gradient
dL_dy = np.random.randn(32, 64)

# Backward pass
dL_dx = layer.backward(dL_dy)
print(f"Input gradient shape: {dL_dx.shape}")    # (32, 128)

# Check the gradient shapes match the parameter shapes
print(f"W grad shape: {layer.grads['W'].shape}") # (128, 64)
print(f"b grad shape: {layer.grads['b'].shape}") # (64,)

# Try with a 3D input (e.g., a sequence)
x_3d = np.random.randn(8, 10, 128)               # (batch, timesteps, features)
y_3d = layer(x_3d)
print(f"3D output shape: {y_3d.shape}")           # (8, 10, 64)
```

## What to read next

- **`neutro/layers/base.md`** — The base class that `Dense` inherits from: learn how `__call__` dispatches between symbolic and eager mode.
- **`neutro/layers/core/dropout.md`** — Another core layer with a very different backward pass (stochastic masking).
- **`neutro/activations/relu.md`** — How the ReLU activation computes its forward pass and gradient.
- **`neutro/activations/softmax.md`** — Why softmax needs a special `gradient_fast` method instead of the element-wise path.
