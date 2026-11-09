"""Neutro: a NumPy-only, Keras-style deep learning library for learning.

Start with ``neutro.models.Sequential`` and ``neutro.layers.Dense``. The
``autograd`` subpackage contains the reverse-mode engine used by the layers.
"""

from . import layers
from .layers import Input
from . import activations
from . import initializers
from . import losses
from . import optimizers
from . import metrics
from . import callbacks
from . import autograd
from .models import Sequential, Model

__all__ = [
    "layers", "activations", "initializers", "losses", "optimizers",
    "metrics", "callbacks", "autograd", "Sequential", "Model",
]
