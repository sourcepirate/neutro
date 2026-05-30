from .base import Loss
from .mse import MeanSquaredError
from .categorical_crossentropy import CategoricalCrossentropy
from .vae_loss import VAELoss

def get(identifier):
    if identifier == 'mse': return MeanSquaredError()
    if identifier == 'categorical_crossentropy': return CategoricalCrossentropy()
    if identifier == 'vae': return VAELoss # Note: VAELoss needs a model instance, so this might be tricky
    if isinstance(identifier, Loss): return identifier
    return identifier
