# Activation Functions

## Theory

Activation functions introduce non-linearity into neural networks. Without them, stacking linear layers would collapse into a single linear transformation.

### ReLU — `neutro/activations/relu.py`

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{ReLU}'(x) = \mathbf{1}_{x > 0}$$

- **Gradient**: 1 for positive inputs, 0 for negative. This causes the "dying ReLU" problem where neurons can get stuck at 0.

### Sigmoid — `neutro/activations/sigmoid.py`

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

- Output range: (0, 1). Used for binary classification or as gating mechanism (LSTM, GRU).
- **Vanishing gradient**: for very large or very small inputs, the gradient approaches 0.

### Tanh — `neutro/activations/tanh.py`

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

$$\tanh'(x) = 1 - \tanh^2(x)$$

- Output range: (-1, 1). Zero-centered, often preferred over sigmoid in hidden layers.

### Softmax — `neutro/activations/softmax.py`

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

- Output: probability distribution over classes.
- **Jacobian-Vector Product** (`gradient_fast`, line 18): computes $y * (\text{grad\_output} - \sum(y * \text{grad\_output}))$ without building the full $N \times N$ Jacobian.

### SiLU — `neutro/activations/silu.py` (Sigmoid Linear Unit)

$$\text{SiLU}(x) = x \cdot \sigma(x)$$

$$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))$$

- Also called Swish. Used in modern architectures (e.g., Llama, GPT).

## Implementation Guide

All activations follow the same pattern:

```python
class ReLU:
    def forward(self, x): ...
    def gradient(self, x): ...        # element-wise gradient
    def gradient_fast(self, x, grad): ...  # fused JVP (optional)
```

- `forward` is used by `Dense` and other layers in the forward pass.
- `gradient` returns the element-wise derivative, which is multiplied by the upstream gradient in `Dense.backward`.
- `gradient_fast` is an optimization used by Softmax to avoid the full Jacobian matrix.

## Usage Example

```python
from neutro.activations import get_activation

relu = get_activation('relu')
x = np.array([-1, 0, 2])
y = relu(x)          # [0, 0, 2]
dy = relu.gradient(x)  # [0, 0, 1]
```

## References

- Nair, V., & Hinton, G. E. (2010). **Rectified Linear Units Improve Restricted Boltzmann Machines**.
- Hendrycks, D., & Gimpel, K. (2016). **Gaussian Error Linear Units (GELUs)**.
- Elfwing, S., Uchibe, E., & Doya, K. (2018). **Sigmoid-weighted linear units for neural network function approximation in reinforcement learning**.
