import numpy as np
from neutro.layers.pooling.maxpooling2d import MaxPooling2D

def test_maxpooling2d():
    layer = MaxPooling2D(pool_size=2)
    x = np.random.rand(2, 4, 4, 1)
    out = layer(x)
    assert out.shape == (2, 2, 2, 1)
    
    grad = layer.backward(np.random.rand(2, 2, 2, 1))
    assert grad.shape == (2, 4, 4, 1)

def test_maxpooling2d_channels_first():
    layer = MaxPooling2D(pool_size=2, data_format='channels_first')
    x = np.random.rand(2, 1, 4, 4)
    out = layer(x)
    assert out.shape == (2, 1, 2, 2)

    grad = layer.backward(np.random.rand(2, 1, 2, 2))
    assert grad.shape == (2, 1, 4, 4)
