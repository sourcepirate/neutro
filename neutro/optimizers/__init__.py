from .base import Optimizer
from .sgd import SGD
from .adam import Adam
from .adamw import AdamW

__all__ = ["Optimizer", "SGD", "Adam", "AdamW", "get"]


def get(identifier):
    """Resolve an optimizer name or instance.

    Keras lets users write ``model.compile(optimizer='adam', ...)``.  This
    helper keeps that promise while still accepting an already-built
    optimizer object.
    """
    if isinstance(identifier, Optimizer):
        return identifier
    if isinstance(identifier, str):
        key = identifier.lower()
        factories = {"sgd": SGD, "adam": Adam, "adamw": AdamW}
        if key in factories:
            return factories[key]()
        raise ValueError(
            f"Unknown optimizer '{identifier}'. "
            f"Expected one of {sorted(factories)} or an Optimizer instance."
        )
    return identifier
