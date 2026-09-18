import numpy as np


class Initializer:
    """Base class for weight initializers."""

    def __call__(self, shape):
        """Return an array of the given shape initialized per this strategy."""
        raise NotImplementedError

    @staticmethod
    def _calculate_fan_in_and_fan_out(shape):
        """Compute fan_in and fan_out for a weight shape.

        Handles 1-D (bias), 2-D (dense) and N-D (convolutional) shapes.
        """
        if len(shape) < 2:
            return shape[0], shape[0]
        if len(shape) == 2:
            return shape[0], shape[1]
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        return fan_in, fan_out
