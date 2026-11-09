class Metric:
    """Base class for all metrics.

    Metrics are for reporting only; they do not participate in backprop.
    """

    def __call__(self, y_true, y_pred):
        """Compute the metric value between targets and predictions."""
        raise NotImplementedError

    def get_name(self) -> str:
        """Return the display name used in ``History`` and logs."""
        raise NotImplementedError
