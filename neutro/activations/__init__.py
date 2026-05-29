from .base import Activation
from .relu import ReLU
from .sigmoid import Sigmoid
from .tanh import Tanh
from .softmax import Softmax

def get(identifier):
    if identifier == 'relu': return ReLU()
    if identifier == 'sigmoid': return Sigmoid()
    if identifier == 'tanh': return Tanh()
    if identifier == 'softmax': return Softmax()
    if isinstance(identifier, Activation): return identifier
    return identifier
