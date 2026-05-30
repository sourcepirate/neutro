from .base import Loss
from .mse import MeanSquaredError
from .categorical_crossentropy import CategoricalCrossentropy
from .vae_loss import VAELoss

def get(identifier):
    if identifier == 'mse': return MeanSquaredError()
    if identifier == 'categorical_crossentropy': return CategoricalCrossentropy()
    if identifier == 'vae':
        raise ValueError("Pass an instantiated VAELoss(model, ...) when compiling a VAE model.")
    if isinstance(identifier, Loss): return identifier
    return identifier
