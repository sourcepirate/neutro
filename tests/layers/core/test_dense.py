import numpy as np
from neutro.layers.core.dense import Dense

def test_dense():
    layer = Dense(10, activation='relu')
    x = np.random.rand(5, 4)
    out = layer(x)
    assert out.shape == (5, 10)
    
    grad = layer.backward(np.random.rand(5, 10))
    assert grad.shape == (5, 4)
    assert 'W' in layer.grads
    assert 'b' in layer.grads
