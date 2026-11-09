from typing import List


def _get_data(param):
    """Unwrap Tensor to ndarray, pass through ndarray."""
    from neutro.autograd import Tensor

    return param.data if isinstance(param, Tensor) else param


class Optimizer:
    """Base class for all optimizers.

    Subclasses implement :meth:`step`, which reads each layer's parameter
    gradients and updates the parameters in place.
    """

    def __init__(self, learning_rate: float = 0.001) -> None:
        self.learning_rate = learning_rate

    @property
    def lr(self) -> float:
        """Learning rate (alias used by callbacks and schedules)."""
        return self.learning_rate

    @lr.setter
    def lr(self, value: float) -> None:
        self.learning_rate = value

    def step(self, layers: List) -> None:
        """Update trainable parameters given a list of layers."""
        raise NotImplementedError
