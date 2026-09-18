from .base import Activation
from .relu import ReLU
from .sigmoid import Sigmoid
from .tanh import Tanh
from .softmax import Softmax
from .silu import SiLU

def get(identifier):
    if identifier is None:
        return None
    if identifier == "relu":
        return ReLU()
    if identifier == "sigmoid":
        return Sigmoid()
    if identifier == "tanh":
        return Tanh()
    if identifier == "softmax":
        return Softmax()
    if identifier == "silu":
        return SiLU()
    if isinstance(identifier, Activation):
        return identifier
    if isinstance(identifier, str):
        raise ValueError(f"Unknown activation identifier '{identifier}'")
    return identifier
