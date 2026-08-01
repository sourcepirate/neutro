class Metric:
    """Base class for all metrics."""

    def __call__(self, y_true, y_pred):
        """Compute the metric value between targets and predictions."""
        raise NotImplementedError

    def get_name(self) -> str:
        """Return the display name of the metric."""
        raise NotImplementedError
