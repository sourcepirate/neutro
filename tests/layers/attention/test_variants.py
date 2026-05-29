import numpy as np
from neutro.layers.attention.mqa import MultiQueryAttention
from neutro.layers.attention.gqa import GroupedQueryAttention

def test_mqa():
    layer = MultiQueryAttention(num_heads=4, key_dim=16)
    x = np.random.rand(2, 5, 16)
    out = layer(x)
    assert out.shape == (2, 5, 16)
    assert layer.backward(np.random.rand(2, 5, 16)) is None

def test_gqa():
    layer = GroupedQueryAttention(num_heads=4, num_groups=2, key_dim=16)
    x = np.random.rand(2, 5, 16)
    out = layer(x)
    assert out.shape == (2, 5, 16)
    assert layer.backward(np.random.rand(2, 5, 16)) is None
