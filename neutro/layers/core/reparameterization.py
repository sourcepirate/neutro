import numpy as np
from ..base import Layer

class Reparameterization(Layer):
    """
    Reparameterization trick for VAE.
    Takes [z_mean, z_log_var] as input and samples z = z_mean + exp(0.5 * z_log_var) * epsilon.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def forward(self, inputs, training=False):
        """
        inputs: list of [z_mean, z_log_var]
        """
        self.z_mean, self.z_log_var = inputs
        
        if not training:
            return self.z_mean
            
        self.epsilon = np.random.normal(size=self.z_mean.shape)
        self.z = self.z_mean + np.exp(0.5 * self.z_log_var) * self.epsilon
        return self.z

    def backward(self, grad_output):
        # grad_output is dL/dz
        # dz/dz_mean = 1
        # dz/dz_log_var = exp(0.5 * z_log_var) * 0.5 * epsilon
        
        grad_mean = grad_output
        grad_log_var = grad_output * np.exp(0.5 * self.z_log_var) * 0.5 * self.epsilon
        
        return [grad_mean, grad_log_var]
