# Initializers

## Theory

Weight initialization is critical for training deep networks. Poor initialization can cause vanishing/exploding gradients. `neutro` implements several strategies.

### Glorot (Xavier) Uniform — `neutro/initializers/glorot.py`

$$W \sim U\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]$$

Recommended for layers with tanh or sigmoid activation.

### He Initialization — `neutro/initializers/he.py`

$$W \sim N\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

Recommended for layers with ReLU activation. Keeps variance of activations constant across layers.

### Constant — `neutro/initializers/constant.py`

$W = c$ for a constant $c$. Used for bias initialization (typically $c=0$).

### Random — `neutro/initializers/random.py`

$$W \sim N(\text{mean}, \text{stddev})$$

## Implementation Guide

All initializers are callable objects:

```python
class GlorotUniform:
    def __call__(self, shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape)
```

They are instantiated in layer `__init__` and called in `build`:

```python
class Dense(Layer):
    def __init__(self, units, kernel_initializer='glorot_uniform', ...):
        self.kernel_initializer = get_initializer(kernel_initializer)

    def build(self, input_shape):
        self.params['W'] = self.kernel_initializer((input_shape[-1], self.units))
```

## References

- Glorot, X., & Bengio, Y. (2010). **Understanding the difficulty of training deep feedforward neural networks**. *AISTATS*.
- He, K., et al. (2015). **Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification**. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
