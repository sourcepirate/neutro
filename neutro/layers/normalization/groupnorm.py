import numpy as np
from ..base import Layer

class GroupNormalization(Layer):
    """
    Group normalization layer (Wu and He, 2018).

    Args:
        groups: Integer, number of groups to separate the channels into.
        epsilon: Small float added to variance to avoid dividing by zero.
    """
    def __init__(self, groups=32, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        # input_shape: (batch, height, width, channels)
        dim = input_shape[-1]
        if dim % self.groups != 0:
            raise ValueError(f"Number of channels ({dim}) must be divisible by groups ({self.groups})")
            
        self.params['gamma'] = np.ones((1, 1, 1, dim))
        self.params['beta'] = np.zeros((1, 1, 1, dim))
        super().build(input_shape)

    def forward(self, x, training=False):
        self.x_shape = x.shape
        batch, h, w, c = x.shape
        g = self.groups
        
        # Reshape to (N, H, W, G, C//G)
        x_reshaped = x.reshape(batch, h, w, g, c // g)
        
        # Calculate mean and var over (H, W, C//G)
        self.mean = np.mean(x_reshaped, axis=(1, 2, 4), keepdims=True)
        self.var = np.var(x_reshaped, axis=(1, 2, 4), keepdims=True)
        
        self.std = np.sqrt(self.var + self.epsilon)
        self.x_centered = x_reshaped - self.mean
        self.x_norm = self.x_centered / self.std
        
        # Reshape back to (N, H, W, C)
        x_norm = self.x_norm.reshape(batch, h, w, c)
        
        return self.params['gamma'] * x_norm + self.params['beta']

    def backward(self, grad_output):
        batch, h, w, c = self.x_shape
        g = self.groups
        m = h * w * (c // g) # elements per group
        
        # d_gamma and d_beta
        self.grads['gamma'] = np.sum(grad_output * self.x_norm.reshape(batch, h, w, c), axis=(0, 1, 2), keepdims=True)
        self.grads['beta'] = np.sum(grad_output, axis=(0, 1, 2), keepdims=True)
        
        # Gradient wrt x_norm
        dx_norm = grad_output * self.params['gamma']
        dx_norm = dx_norm.reshape(batch, h, w, g, c // g)
        
        # Standard BN-like backward applied per group
        # This is a bit complex but follows the same logic as BN
        sum_dx_norm = np.sum(dx_norm, axis=(1, 2, 4), keepdims=True)
        sum_dx_norm_x_norm = np.sum(dx_norm * self.x_norm, axis=(1, 2, 4), keepdims=True)
        
        dx = (1.0 / m) / self.std * (
            m * dx_norm - sum_dx_norm - self.x_norm * sum_dx_norm_x_norm
        )
        
        return dx.reshape(batch, h, w, c)
