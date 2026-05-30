import numpy as np
from ..base import Layer

class TimeEmbedding(Layer):
    """
    Sinusoidal time embeddings for Diffusion models.
    Converts a batch of timesteps (batch_size, 1) into (batch_size, dim).
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.last_t = None

    def build(self, input_shape):
        super().build(input_shape)

    def forward(self, t, training=False):
        """
        t: array of shape (batch_size,) or (batch_size, 1)
        """
        if t.ndim == 2:
            t = t.flatten()
        self.last_t = t
            
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = np.exp(np.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = np.concatenate([np.sin(embeddings), np.cos(embeddings)], axis=1)
        
        if self.dim % 2 == 1:
            embeddings = np.pad(embeddings, ((0, 0), (0, 1)))
            
        return embeddings

    def backward(self, grad_output):
        # Timesteps are usually fixed/non-trainable inputs, 
        # so gradient wrt t is typically not needed or 0.
        return np.zeros_like(self.last_t)
