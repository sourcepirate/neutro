import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}

    def step(self, layers):
        for layer in layers:
            if not getattr(layer, 'trainable', True):
                continue
            for param_name, param_value in layer.params.items():
                grad = layer.grads[param_name]
                key = (id(layer), param_name)
                
                if self.momentum > 0:
                    if key not in self.velocities:
                        self.velocities[key] = np.zeros_like(param_value)
                    
                    v = self.velocities[key]
                    v_next = self.momentum * v - self.learning_rate * grad
                    self.velocities[key] = v_next
                    
                    if self.nesterov:
                        layer.params[param_name] += self.momentum * v_next - self.learning_rate * grad
                    else:
                        layer.params[param_name] += v_next
                else:
                    layer.params[param_name] -= self.learning_rate * grad
