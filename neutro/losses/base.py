import numpy as np


class Loss:
    """Base class for all losses."""

    def __call__(self, y_true, y_pred):
        """Compute the loss value between targets and predictions."""
        raise NotImplementedError

    def gradient(self, y_true, y_pred) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. the predictions."""
        raise NotImplementedError
