import numpy as np
from ..base import Layer
from ..core.dense import Dense
from ...activations.silu import SiLU
from neutro.autograd import Tensor


class TimeMLPEmbedding(Layer):
    """Sinusoidal timestep embedding + MLP (DiT-style).

    t: (B,) int timesteps -> (B, dim).
    """

    def __init__(self, dim, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.fc1 = Dense(self.hidden_dim)
        self.fc2 = Dense(dim)
        self.silu = SiLU()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

    def build(self, input_shape):
        self.fc1.build((input_shape[0], self.dim))
        self.fc2.build((input_shape[0], self.hidden_dim))
        super().build(input_shape)

    def _sinusoid(self, t):
        t = np.asarray(t).reshape(-1)
        half = self.dim // 2
        inv = np.exp(-np.arange(half) * (np.log(10000.0) / max(half, 1)))
        emb = t[:, None] * inv[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
        if self.dim % 2 == 1:
            emb = np.pad(emb, ((0, 0), (0, 1)))
        return Tensor(emb)

    def forward(self, t, training=False):
        h = self._sinusoid(t)
        return self.fc2(self.silu(self.fc1(h, training)), training)
