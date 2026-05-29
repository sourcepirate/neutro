import numpy as np
import pytest
from neutro.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

def test_global_average_pooling():
    batch, h, w, c = 2, 4, 4, 3
    layer = GlobalAveragePooling2D()
    
    inputs = np.random.randn(batch, h, w, c)
    out = layer.forward(inputs)
    
    assert out.shape == (batch, c)
    assert np.allclose(out[0, 0], np.mean(inputs[0, :, :, 0]))

def test_global_max_pooling():
    batch, h, w, c = 2, 4, 4, 3
    layer = GlobalMaxPooling2D()
    
    inputs = np.random.randn(batch, h, w, c)
    out = layer.forward(inputs)
    
    assert out.shape == (batch, c)
    assert out[0, 0] == np.max(inputs[0, :, :, 0])

def test_global_pooling_backward():
    batch, h, w, c = 2, 4, 4, 3
    avg_layer = GlobalAveragePooling2D()
    max_layer = GlobalMaxPooling2D()
    
    inputs = np.random.randn(batch, h, w, c)
    
    # Avg
    avg_layer.forward(inputs)
    grad_avg = avg_layer.backward(np.random.randn(batch, c))
    assert grad_avg.shape == inputs.shape
    
    # Max
    max_layer.forward(inputs)
    grad_max = max_layer.backward(np.random.randn(batch, c))
    assert grad_max.shape == inputs.shape
