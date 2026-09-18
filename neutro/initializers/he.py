import numpy as np

from .base import Initializer


class HeNormal(Initializer):
    """He normal initializer."""

    def __call__(self, shape):
        fan_in, _ = self._calculate_fan_in_and_fan_out(shape)
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, shape)
