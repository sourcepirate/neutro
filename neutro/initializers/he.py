import numpy as np
from .base import Initializer

class HeNormal(Initializer):
    def __call__(self, shape):
        fan_in, _ = self._calculate_fan_in_and_fan_out(shape)
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, shape)

    def _calculate_fan_in_and_fan_out(self, shape):
        if len(shape) < 2:
            return shape[0], shape[0]
        if len(shape) == 2:
            return shape[0], shape[1]
        else:
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
            return fan_in, fan_out
