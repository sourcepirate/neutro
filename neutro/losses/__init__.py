from .base import Loss
from .mse import MeanSquaredError
from .categorical_crossentropy import CategoricalCrossentropy

def get(identifier):
    if identifier == 'mse': return MeanSquaredError()
    if identifier == 'categorical_crossentropy': return CategoricalCrossentropy()
    if isinstance(identifier, Loss): return identifier
    return identifier
