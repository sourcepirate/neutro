import numpy as np
from neutro.layers.attention.mqa import MultiQueryAttention
from neutro.layers.attention.gqa import GroupedQueryAttention
from neutro.layers.attention.mla import MultiHeadLatentAttention

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

def test_mla():
    layer = MultiHeadLatentAttention(num_heads=4, head_dim=4, latent_dim=16, kv_latent_dim=8)
    layer.build((2, 5, 16))
    x = np.random.rand(2, 5, 16)
    out = layer(x)
    assert out.shape == (2, 5, 16)
    grad = layer.backward(np.random.rand(2, 5, 16))
    assert grad.shape == (2, 5, 16)
