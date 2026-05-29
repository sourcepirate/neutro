import numpy as np
from neutro.layers.core.flatten import Flatten

def test_flatten():
    layer = Flatten()
    x = np.random.rand(2, 3, 4)
    out = layer(x)
    assert out.shape == (2, 12)
    
    grad = layer.backward(np.random.rand(2, 12))
    assert grad.shape == (2, 3, 4)
