from .base import Loss
from .mse import MeanSquaredError
from .categorical_crossentropy import CategoricalCrossentropy
from .sparse_categorical_crossentropy import SparseCategoricalCrossentropy
from .masked_sparse_ce import MaskedSparseCrossentropy
from .vae_loss import VAELoss

__all__ = [
    "Loss", "MeanSquaredError", "CategoricalCrossentropy",
    "SparseCategoricalCrossentropy", "MaskedSparseCrossentropy",
    "VAELoss", "get",
]

def get(identifier):
    if identifier == "mse":
        return MeanSquaredError()
    if identifier == "categorical_crossentropy":
        return CategoricalCrossentropy()
    if identifier == "sparse_categorical_crossentropy":
        return SparseCategoricalCrossentropy()
    if identifier == "vae":
        raise ValueError("Pass an instantiated VAELoss(model, ...) when compiling a VAE model.")
    if isinstance(identifier, Loss):
        return identifier
    if isinstance(identifier, str):
        raise ValueError(f"Unknown loss identifier '{identifier}'")
    return identifier
