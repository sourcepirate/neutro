import numpy as np
from .precision import Precision
from .recall import Recall
from .base import Metric

class F1Score(Metric):
    def __call__(self, y_true, y_pred):
        p = Precision()(y_true, y_pred)
        r = Recall()(y_true, y_pred)
        return 2 * p * r / (p + r + 1e-15)
    def get_name(self):
        return "f1_score"
