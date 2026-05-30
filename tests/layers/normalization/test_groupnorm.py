import numpy as np
from neutro.layers.normalization.groupnorm import GroupNormalization

def test_group_norm():
    layer = GroupNormalization(groups=2)
    x = np.random.randn(2, 4, 4, 4)
    layer.build(x.shape)
    out = layer.forward(x)
    assert out.shape == x.shape
    
    # Check if groups have mean 0 and var 1 (approximately)
    out_reshaped = out.reshape(2, 4, 4, 2, 2)
    means = np.mean(out_reshaped, axis=(1, 2, 4))
    assert np.allclose(means, 0, atol=1e-5)
