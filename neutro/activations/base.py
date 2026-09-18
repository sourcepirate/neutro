class Activation:
    """Base class for activation functions."""

    def __call__(self, x):
        """Apply the activation function."""
        raise NotImplementedError

    def gradient(self, x):
        """Compute the derivative of the activation function."""
        raise NotImplementedError
