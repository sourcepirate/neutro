class Initializer:
    """Base class for weight initializers."""

    def __call__(self, shape):
        """Return an array of the given shape initialized per this strategy."""
        raise NotImplementedError
