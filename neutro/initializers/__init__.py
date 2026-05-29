from .base import Initializer
from .constant import Zeros, Ones
from .random import RandomNormal
from .glorot import GlorotUniform
from .he import HeNormal

def get(identifier):
    if identifier == 'zeros': return Zeros()
    if identifier == 'ones': return Ones()
    if identifier == 'glorot_uniform': return GlorotUniform()
    if identifier == 'he_normal': return HeNormal()
    if isinstance(identifier, Initializer): return identifier
    return identifier
