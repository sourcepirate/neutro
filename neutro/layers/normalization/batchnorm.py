import numpy as np
from ..base import Layer

class BatchNormalization(Layer):
    """
    Batch normalization layer (Ioffe and Szegedy, 2014).

    Args:
        momentum: Momentum for the moving average.
        epsilon: Small float added to variance to avoid dividing by zero.
    """
    def __init__(self, momentum=0.99, epsilon=1e-3):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None

    def build(self, input_shape):
        dim = input_shape[-1]
        self.params['gamma'] = np.ones(dim)
        self.params['beta'] = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        super().build(input_shape)

    def forward(self, x, training=False):
        if training:
            mean = np.mean(x, axis=tuple(range(len(x.shape)-1)))
            var = np.var(x, axis=tuple(range(len(x.shape)-1)))
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.x_centered = x - mean
            self.std = np.sqrt(var + self.epsilon)
            self.x_norm = self.x_centered / self.std
        else:
            x_centered = x - self.running_mean
            std = np.sqrt(self.running_var + self.epsilon)
            self.x_norm = x_centered / std
            
        return self.params['gamma'] * self.x_norm + self.params['beta']

    def backward(self, grad_output):
        # Implementation of BN backward is a bit involved
        gamma = self.params['gamma']
        batch_size = np.prod(grad_output.shape[:-1])
        
        # d_gamma and d_beta
        self.grads['gamma'] = np.sum(grad_output * self.x_norm, axis=tuple(range(len(grad_output.shape)-1)))
        self.grads['beta'] = np.sum(grad_output, axis=tuple(range(len(grad_output.shape)-1)))
        
        dx_norm = grad_output * gamma
        dx = (1. / batch_size) / self.std * (
            batch_size * dx_norm - np.sum(dx_norm, axis=tuple(range(len(grad_output.shape)-1))) -
            self.x_norm * np.sum(dx_norm * self.x_norm, axis=tuple(range(len(grad_output.shape)-1)))
        )
        return dx
