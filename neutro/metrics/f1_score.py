from .base import Metric
from .precision import Precision
from .recall import Recall


class F1Score(Metric):
    """F1 score — harmonic mean of precision and recall."""

    def __init__(self):
        self._precision = Precision()
        self._recall = Recall()

    def __call__(self, y_true, y_pred):
        p = self._precision(y_true, y_pred)
        r = self._recall(y_true, y_pred)
        return 2 * p * r / (p + r + 1e-15)

    def get_name(self):
        return "f1_score"
