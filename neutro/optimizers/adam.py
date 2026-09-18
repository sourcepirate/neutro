import numpy as np

from neutro.autograd import Tensor

from .base import Optimizer, _get_data

DEFAULT_LR = 0.001
DEFAULT_BETA_1 = 0.9
DEFAULT_BETA_2 = 0.999
DEFAULT_EPSILON = 1e-7


class Adam(Optimizer):
    def __init__(self, learning_rate=DEFAULT_LR, beta_1=DEFAULT_BETA_1,
                 beta_2=DEFAULT_BETA_2, epsilon=DEFAULT_EPSILON):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not getattr(layer, 'trainable', True):
                continue
            for param_name, param_value in layer.params.items():
                grad = (param_value.grad
                        if isinstance(param_value, Tensor) and param_value.grad is not None
                        else layer.grads.get(param_name))
                if grad is None:
                    continue
                self._apply_step(layer, param_name, param_value, grad)

    def _apply_step(self, layer, param_name, param_value, grad):
        param_data = _get_data(param_value)
        key = (id(layer), param_name)

        if key not in self.m:
            self.m[key] = np.zeros_like(param_data)
            self.v[key] = np.zeros_like(param_data)

        self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
        self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta_1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta_2 ** self.t)

        param_data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
