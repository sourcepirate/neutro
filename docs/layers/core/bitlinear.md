# BitLinear Layer

## What does this layer do?

`BitLinear` is the core building block of **BitNet** — a 1-bit Transformer architecture. It replaces a standard `Dense` / `nn.Linear` layer with one that:

1. **Quantizes weights** to 1-bit (`{-1, +1}` in b1 mode) or 1.58-bit (`{-1, 0, +1}` in b1.58 mode)
2. **Quantizes activations** to 8-bit integers via absmax quantization
3. **SubLN** — applies LayerNorm before activation quantization to stabilize variance
4. **STE (Straight-Through Estimator)** — gradients bypass the non-differentiable quantization, flowing as if it were a standard linear layer

Paper: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## The math

### Weight quantization (b1 mode)

$$W_q = \text{Sign}(W - \alpha), \quad \beta = \frac{1}{nm}\sum_{ij}|W_{ij}|$$

where $\alpha = \frac{1}{nm}\sum_{ij}W_{ij}$ (mean centralization) and $\beta$ (a scaling factor) is the mean absolute value of the weights.

### Weight quantization (b1.58 mode)

$$W_{\text{scaled}} = \frac{W}{\gamma_w}, \quad W_q = \text{Clip}(\text{round}(W_{\text{scaled}}), -1, 1), \quad \beta = \frac{1}{nm}\sum_{ij}|W_{ij}|$$

where $\gamma_w = \frac{1}{nm}\sum_{ij}|W_{ij}|$ (the absmean).

### Activation quantization (absmax)

$$\tilde{x} = \text{Clip}(\text{round}(x \times \frac{Q_b}{\gamma}), -Q_b + 1, Q_b - 1), \quad \gamma = ||x||_{\infty}$$

where $Q_b = 2^{b-1}$ (for 8-bit: $Q_8 = 128$).

### BitLinear forward

$$y = \tilde{W}\tilde{x} = W_q \cdot \tilde{x} \times \frac{\beta \gamma}{Q_b}$$

The full flow:
1. **SubLN**: $x_{\text{norm}} = \text{LayerNorm}(x)$
2. **Activation quant**: $\tilde{x} = \text{Quant}(x_{\text{norm}})$ — produces integer values in $[-128, 127]$
3. **Weight quant**: $W_q = \text{Quant}_W(W)$ — produces values in $\{-1, +1\}$ or $\{-1, 0, +1\}$
4. **Matrix multiply + dequantize**: $y = (x_q @ W_q^T) \times \frac{\beta \gamma}{Q_b}$

### Straight-Through Estimator (backward)

During backward, quantization is treated as identity:

$$\frac{\partial L}{\partial W} \approx x^T \cdot \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y} \cdot W^T$$

The full-precision weights $W$ receive gradients as if no quantization occurred. Over time, they "learn" to produce good quantized versions.

## Walking through the code

### File: `neutro/layers/core/bitlinear.py`

### Module-level quantization functions

```python
def weight_quantize_b1(W, eps=1e-6):
    alpha = np.mean(W)
    W_centered = W - alpha
    W_bin = np.where(W_centered > 0, 1.0, -1.0)
    beta = np.mean(np.abs(W)) + eps
    return W_bin, beta
```

- **Line 6**: $\alpha = \text{mean}(W)$. Centralizes weights to zero-mean before binarization.
- **Line 7-8**: Sign function: $W_q = \text{Sign}(W - \alpha)$. Every value becomes exactly +1 or -1.
- **Line 9**: $\beta = \text{mean}(|W|)$. Scaling factor that minimizes L2 error between real-valued and binarized weights.
- **Line 10**: Returns $W_q \in \{-1, +1\}^{n \times m}$ and $\beta \in \mathbb{R}$.

```python
def weight_quantize_b158(W, eps=1e-6):
    gamma = np.mean(np.abs(W)) + eps
    W_scaled = W / gamma
    W_tern = np.clip(np.round(W_scaled), -1, 1)
    beta = np.mean(np.abs(W)) + eps
    return W_tern, beta
```

- **Line 13**: $\gamma_w = \text{mean}(|W|)$. The absmean scale.
- **Line 14**: Normalize by absmean: $W / \gamma_w$.
- **Line 15**: Round to nearest integer and clip to $[-1, 1]$. Result: $W_q \in \{-1, 0, +1\}^{n \times m}$.
- **Line 16**: $\beta = \text{mean}(|W|)$, same L2-minimizing scale as b1.
- **Line 17**: Returns ternary weights and scale.

```python
def activation_quantize(x, bits=8, per_token=False, eps=1e-6):
    Q_b = 2 ** (bits - 1)
    if per_token:
        abs_max = np.max(np.abs(x), axis=-1, keepdims=True)
        abs_max = np.clip(abs_max, eps, None)
        gamma = abs_max
    else:
        gamma = np.max(np.abs(x))
        if gamma < eps:
            gamma = eps
    x_scaled = x * Q_b / gamma
    x_quant = np.clip(np.round(x_scaled), -Q_b + 1, Q_b - 1)
    return x_quant, gamma
```

- **Line 20**: $Q_b = 2^{b-1}$. For 8-bit: 128.
- **Lines 21-24**: Per-token quantization (used during inference). Each token (row) gets its own $\gamma$.
- **Lines 25-28**: Per-tensor quantization (used during training). A single $\gamma = ||x||_{\infty}$ for the whole tensor.
- **Line 30**: Scale by $Q_b / \gamma$ and round.
- **Line 31**: Clip to $[-127, 127]$ for 8-bit.

### Class: `BitLinear`

#### `__init__` (bitlinear.py:35)

```python
class BitLinear(Layer):
    def __init__(self, units, mode='b1.58', activation_bits=8,
                 use_bias=False, per_token=False,
                 kernel_initializer='glorot_uniform', **kwargs):
```

- **`mode`**: `'b1'` (binary $\{-1, +1\}$) or `'b1.58'` (ternary $\{-1, 0, +1\}$).
- **`activation_bits`**: Bit-width for activation quantization (default 8).
- **`use_bias`**: BitNet omits biases; kept for API compatibility.
- **`per_token`**: Per-token vs per-tensor activation quantization.

#### `build` (bitlinear.py:45)

```python
def build(self, input_shape):
    self.input_dim = input_shape[-1]
    self.params['W'] = self.kernel_initializer((self.input_dim, self.units))
    if self.use_bias:
        self.params['b'] = np.zeros((self.units,))
    self.params['gamma_ln'] = np.ones(self.input_dim)
    self.params['beta_ln'] = np.zeros(self.input_dim)
```

Two sets of parameters:

| Parameter | Shape | Role |
|---|---|---|
| `W` | `(input_dim, units)` | Full-precision weight (trained; quantized in forward) |
| `gamma_ln` | `(input_dim,)` | SubLN scale |
| `beta_ln` | `(input_dim,)` | SubLN shift |

#### `forward` (bitlinear.py:65)

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    x_norm = self._layernorm_forward(inputs)           # SubLN
    x_q, self.gamma = self._activation_quantize(x_norm) # Activation quant
    W_q, self.beta = self._weight_quantize(self.params['W']) # Weight quant
```

**Shape walkthrough** (batch=2, seq=8, dim=32, units=64):
- `inputs`: `(2, 8, 32)`
- `x_norm`: `(2, 8, 32)` — LayerNorm (no shape change)
- `x_q`: `(2, 8, 32)` — quantized to $[-128, 127]$ integers
- `W_q`: `(32, 64)` — quantized to $\{-1, 0, +1\}$
- `self.gamma`: scalar (per-tensor) or `(2, 8, 1)` (per-token)
- `self.beta`: scalar

```python
    self.Q_b = 2 ** (self.activation_bits - 1)
    self.deq_scale = self.beta * self.gamma / self.Q_b
    y = (x_q @ W_q.T) * self.deq_scale
```

- `x_q @ W_q.T`: `(2, 8, 32) @ (64, 32).T = (2, 8, 32) @ (32, 64) → (2, 8, 64)` 🔍 INT-INT matmul
- `* self.deq_scale`: element-wise broadcast — dequantizes back to full precision
- Output: `(2, 8, 64)`

#### `backward` (bitlinear.py:82)

```python
def backward(self, grad_output):
    inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])
    grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
    self.grads['W'] = inputs_flat.T @ grad_flat
    dx = grad_flat @ self.params['W'].T
```

**Shape walkthrough**:
- `grad_output`: `(2, 8, 64)`
- `grad_flat`: `(16, 64)` — flattened batch and sequence
- `inputs_flat.T @ grad_flat`: `(32, 16) @ (16, 64) → (32, 64)` — **gradient for W** ✅ same shape as `self.params['W']`
- `grad_flat @ self.params['W'].T`: `(16, 64) @ (64, 32) → (16, 32)` — **gradient for input** ✅

This is STE in action: quantization is completely bypassed in the backward pass. The full-precision `W` receives gradients as if the forward pass used standard matmul.

```python
    dx = self._layernorm_backward(dx)
    return dx
```

Standard LayerNorm backward, same as `neutro/layers/normalization/layernorm.py`.

### Key insight: why STE works

The backward pass would normally require:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_q} \cdot \frac{\partial W_q}{\partial W}$$

But $\frac{\partial W_q}{\partial W}$ is zero almost everywhere (rounding/sign have zero derivative). STE replaces this with identity:

$$\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial W_q}$$

This means the full-precision weights "see" standard gradients, and the quantization acts only as a regularizer in the forward pass.
