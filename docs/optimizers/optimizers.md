# Optimizers

An optimizer turns `layer.grads` into parameter updates. All optimizers inherit
from `neutro.optimizers.base.Optimizer` and implement `step(layers)`.

Source: `neutro/optimizers/`.

`model.compile` accepts either an instance or a Keras-style name:

```python
from neutro.optimizers import Adam, SGD, AdamW

model.compile(optimizer=Adam(learning_rate=0.001),...)
model.compile(optimizer="adam",...) # same defaults
model.compile(optimizer="sgd",...)
model.compile(optimizer="adamw",...)
```

---

## SGD — `neutro/optimizers/sgd.py`

Vanilla update:

```text
w = w - learning_rate * grad
```

With momentum:

```text
v = momentum * v - learning_rate * grad
w = w + v
```

With Nesterov momentum:

```text
v = momentum * v - learning_rate * grad
w = w + momentum * v - learning_rate * grad
```

`SGD(learning_rate=0.01, momentum=0.9, nesterov=True)`.

The optimizer keys its velocity per `(id(layer), param_name)`, so layers can be
added or removed without corrupting other state.

---

## Adam — `neutro/optimizers/adam.py`

```text
m = beta_1 * m + (1 - beta_1) * grad
v = beta_2 * v + (1 - beta_2) * grad^2
m_hat = m / (1 - beta_1^t)
v_hat = v / (1 - beta_2^t)
w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

Defaults: `learning_rate=0.001`, `beta_1=0.9`, `beta_2=0.999`,
`epsilon=1e-7`. `t` increments once per `step` call (not per parameter).

---

## AdamW — `neutro/optimizers/adamw.py`

AdamW applies decoupled weight decay before the Adam update:

```text
w = w - learning_rate * weight_decay * w
# then the ordinary Adam update
```

This differs from adding `weight_decay * w` to the gradient (classic Adam +
L2), where the decay is scaled by the adaptive denominator. See
[Loshchilov & Hutter (2017)](https://arxiv.org/abs/1711.05101).

Defaults: `weight_decay=0.01`.

---

## Learning-rate schedules — `neutro/optimizers/schedules.py`

These are standalone callables; they do not plug into an optimizer
automatically:

```python
from neutro.optimizers.schedules import ExponentialDecay

schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96,
)
lr = schedule(step=100) # a float
```

Available:

- `ExponentialDecay`: `lr = lr0 * decay_rate ** (step / decay_steps)`
- `InverseTimeDecay`: `lr = lr0 / (1 + decay_rate * step / decay_steps)`

Both support `staircase=True` to floor the exponent.

To apply a schedule during training, use the `LearningRateScheduler` callback
(`neutro/callbacks/lr_scheduler.py`):

```python
from neutro.callbacks import LearningRateScheduler

def schedule(epoch, current_lr):
    return current_lr * 0.9

model.fit(x, y, epochs=10, callbacks=[LearningRateScheduler(schedule)])
```

Or set `optimizer.lr` manually between epochs.

---

## Adding a new optimizer

Implement `step(layers)` and update each parameter in `layer.params`:

```python
from neutro.autograd import Tensor
from neutro.optimizers.base import Optimizer, _get_data

class MyOptimizer(Optimizer):
    def step(self, layers):
        for layer in layers:
            if not getattr(layer, "trainable", True):
                continue
            for name, param in layer.params.items():
                grad = param.grad if isinstance(param, Tensor) else layer.grads.get(name)
                if grad is None:
                    continue
                _get_data(param)[...] -= self.learning_rate * grad
```

`_get_data` unwraps a `Tensor` to its NumPy array so the update is in place.

---

## References

- Kingma, D. P., & Ba, J. (2014). **Adam: A Method for Stochastic Optimization**. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Loshchilov, I., & Hutter, F. (2017). **Decoupled Weight Decay Regularization**. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Sutskever, I., et al. (2013). **On the importance of initialization and momentum in deep learning**. *ICML*.
