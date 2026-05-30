import numpy as np
from neutro.layers.core.merging import Add, Concatenate

def test_add_layer():
    layer = Add()
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[5, 6], [7, 8]])
    out = layer.forward([x1, x2])
    assert np.array_equal(out, x1 + x2)
    
    grad = layer.backward(np.ones_like(x1))
    assert len(grad) == 2
    assert np.array_equal(grad[0], np.ones_like(x1))
    assert np.array_equal(grad[1], np.ones_like(x2))

def test_concatenate_layer():
    layer = Concatenate(axis=-1)
    x1 = np.random.randn(2, 4, 4, 8)
    x2 = np.random.randn(2, 4, 4, 16)
    layer.build([x1.shape, x2.shape])
    out = layer.forward([x1, x2])
    assert out.shape == (2, 4, 4, 24)
    
    grad = layer.backward(np.ones_like(out))
    assert len(grad) == 2
    assert grad[0].shape == x1.shape
    assert grad[1].shape == x2.shape
