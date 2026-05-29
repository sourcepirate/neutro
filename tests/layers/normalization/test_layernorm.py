import numpy as np
from neutro.layers.normalization.layernorm import LayerNormalization

def test_layernorm():
    layer = LayerNormalization()
    x = np.random.rand(2, 5, 16)
    out = layer(x)
    assert out.shape == (2, 5, 16)
    
    grad = layer.backward(np.random.rand(2, 5, 16))
    assert grad.shape == (2, 5, 16)
