import numpy as np

from .base import Initializer


class GlorotUniform(Initializer):
    """Glorot / Xavier uniform initializer."""

    def __call__(self, shape):
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(shape)
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
