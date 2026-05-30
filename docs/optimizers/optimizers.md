# Optimizers

All optimizers update model parameters using gradients. They inherit from `neutro/optimizers/base.py`:

```python
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layers):  # Called after backward
        for layer in layers:
            for param_name, param_value in layer.params.items():
                grad = layer.grads[param_name]
                param_value -= self.learning_rate * grad  # Vanilla SGD
```

## SGD with Momentum — `neutro/optimizers/sgd.py`

$$v_t = \mu v_{t-1} + \alpha g_t$$
$$w_{t+1} = w_t - v_t$$

Nesterov variant: $w_{t+1} = w_t + \mu v_t - \alpha g_t$

## Adam — `neutro/optimizers/adam.py`

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w_{t+1} = w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Maintains per-parameter `m` (momentum) and `v` (velocity) states, keyed by `(id(layer), param_name)`.

## AdamW — `neutro/optimizers/adamw.py`

Same as Adam but with **decoupled weight decay**:

$$w_{t+1} = w_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_t \right)$$

The weight decay $\lambda w_t$ is applied separately from the adaptive gradient, improving generalization.

## Learning Rate Schedules — `neutro/optimizers/schedules.py`

Provides:

- **ExponentialDecay**: $\alpha_t = \alpha_0 \cdot r^{t/s}$
- **InverseTimeDecay**: $\alpha_t = \alpha_0 / (1 + k \cdot t/s)$

Used as `learning_rate` argument: `Adam(learning_rate=ExponentialDecay(0.001, 100, 0.96))`.

## Usage Example

```python
from neutro.optimizers import Adam, SGD
from neutro.optimizers.schedules import ExponentialDecay

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.001)
adamw = AdamW(learning_rate=3e-4, weight_decay=0.01)

schedule = ExponentialDecay(initial_lr=0.001, decay_steps=1000, decay_rate=0.96)
custom = Adam(learning_rate=schedule)
```

## References

- Kingma, D. P., & Ba, J. (2014). **Adam: A Method for Stochastic Optimization**. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Loshchilov, I., & Hutter, F. (2017). **Decoupled Weight Decay Regularization**. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Sutskever, I., et al. (2013). **On the importance of initialization and momentum in deep learning**. *ICML*.
