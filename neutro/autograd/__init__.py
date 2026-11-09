"""Reverse-mode automatic differentiation built from NumPy.

``Tensor`` stores data + grad. ``GradientTape`` records operations and computes
gradients. See ``docs/autograd/`` for the walkthroughs.
"""

from .tensor import Tensor, as_tensor
from .tape import GradientTape
from .function import Function
from . import ops
from . import utils

__all__ = ["Tensor", "as_tensor", "GradientTape", "Function", "ops", "utils"]
