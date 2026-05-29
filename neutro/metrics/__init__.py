from .base import Metric
from .accuracy import Accuracy
from .precision import Precision
from .recall import Recall
from .f1_score import F1Score

def get(identifier):
    if identifier == 'accuracy': return Accuracy()
    if identifier == 'precision': return Precision()
    if identifier == 'recall': return Recall()
    if identifier == 'f1_score': return F1Score()
    if isinstance(identifier, Metric): return identifier
    return identifier
