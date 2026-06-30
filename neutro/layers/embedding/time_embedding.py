import numpy as np
from ..base import Layer
from neutro.autograd import Tensor as AutoTensor


class TimeEmbedding(Layer):
    """
    Sinusoidal time embeddings (positional encoding for diffusion timesteps).

    PE(t)[i] = sin(t / 10000^{2i/d})   for i even
               cos(t / 10000^{2i/d})   for i odd

    Input:  (B,) or (B, 1)  — timestep indices
    Output: (B, D)          — sinusoidal embedding vectors
    No trainable parameters.
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

    def forward(self, t, training=False):
        if t.ndim == 2:
            t = t.flatten()
        half_dim = self.dim // 2
        inv_freq = np.exp(-np.arange(half_dim) * (np.log(10000.0) / half_dim))
        emb = t[:, None] * inv_freq[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
        if self.dim % 2 == 1:
            emb = np.pad(emb, ((0, 0), (0, 1)))
        return AutoTensor(emb)
