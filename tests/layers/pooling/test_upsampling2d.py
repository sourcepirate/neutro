import numpy as np
from neutro.layers.pooling.upsampling2d import UpSampling2D

def test_upsampling2d():
    layer = UpSampling2D(size=(2, 2))
    x = np.array([[[[1], [2]], [[3], [4]]]]) # (1, 2, 2, 1)
    out = layer.forward(x)
    assert out.shape == (1, 4, 4, 1)
    assert out[0, 0, 0, 0] == 1
    assert out[0, 0, 1, 0] == 1
    assert out[0, 1, 0, 0] == 1
    
    grad = layer.backward(np.ones_like(out))
    assert grad.shape == x.shape
    assert grad[0, 0, 0, 0] == 4 # 2x2 sum
