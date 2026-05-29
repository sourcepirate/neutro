import numpy as np
from .base import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
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
                grad = layer.grads[param_name]
                key = (id(layer), param_name)
                
                if key not in self.m:
                    self.m[key] = np.zeros_like(param_value)
                    self.v[key] = np.zeros_like(param_value)
                
                self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
                self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad**2)
                
                m_hat = self.m[key] / (1 - self.beta_1**self.t)
                v_hat = self.v[key] / (1 - self.beta_2**self.t)
                
                layer.params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
