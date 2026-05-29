import numpy as np
from neutro.layers.attention.mha import MultiHeadAttention

def test_mha():
    layer = MultiHeadAttention(num_heads=2, key_dim=16)
    x = np.random.rand(2, 5, 16)
    out = layer(x)
    assert out.shape == (2, 5, 16)
    
    grad = layer.backward(np.random.rand(2, 5, 16))
    assert grad.shape == (2, 5, 16)
