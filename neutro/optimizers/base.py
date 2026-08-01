from typing import List


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, learning_rate: float = 0.001) -> None:
        self.learning_rate = learning_rate

    @property
    def lr(self) -> float:
        return self.learning_rate

    @lr.setter
    def lr(self, value: float) -> None:
        self.learning_rate = value

    def step(self, layers: List) -> None:
        """Update trainable parameters given a list of layers."""
        raise NotImplementedError
