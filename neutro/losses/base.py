import numpy as np


class Loss:
    """Base class for all losses.

    A loss computes a scalar value in ``__call__`` and its derivative with
    respect to ``y_pred`` in ``gradient``.
    """

    def __call__(self, y_true, y_pred):
        """Compute the loss value between targets and predictions."""
        raise NotImplementedError

    def gradient(self, y_true, y_pred) -> np.ndarray:
        """Return ``dL/dy_pred``, an array with the same shape as ``y_pred``."""
        raise NotImplementedError
