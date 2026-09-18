import numpy as np

from neutro.autograd import Tensor

from .base import Optimizer, _get_data

DEFAULT_LR = 0.01


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and Nesterov."""

    def __init__(self, learning_rate: float = DEFAULT_LR, momentum: float = 0.0,
                 nesterov: bool = False) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}

    def step(self, layers) -> None:
        for layer in layers:
            if not getattr(layer, 'trainable', True):
                continue
            for param_name, param_value in layer.params.items():
                grad = (param_value.grad
                        if isinstance(param_value, Tensor) and param_value.grad is not None
                        else layer.grads.get(param_name))
                if grad is None:
                    continue
                param_data = _get_data(param_value)
                key = (id(layer), param_name)

                if self.momentum > 0:
                    if key not in self.velocities:
                        self.velocities[key] = np.zeros_like(param_data)
                    v_next = self.momentum * self.velocities[key] - self.learning_rate * grad
                    self.velocities[key] = v_next
                    if self.nesterov:
                        param_data += self.momentum * v_next - self.learning_rate * grad
                    else:
                        param_data += v_next
                else:
                    param_data -= self.learning_rate * grad
