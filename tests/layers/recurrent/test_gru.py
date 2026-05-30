import numpy as np
import pytest
from neutro.layers.recurrent.gru import GRU
from neutro.models.base_model import Sequential

def test_gru_shape():
    batch, timesteps, features = 4, 5, 8
    units = 16
    x = np.random.randn(batch, timesteps, features)
    
    # Test return_sequences=False
    layer = GRU(units, return_sequences=False)
    layer.build(x.shape)
    out = layer.forward(x)
    assert out.shape == (batch, units)
    
    # Test return_sequences=True
    layer = GRU(units, return_sequences=True)
    layer.build(x.shape)
    out = layer.forward(x)
    assert out.shape == (batch, timesteps, units)

def test_gru_backward():
    batch, timesteps, features = 2, 3, 4
    units = 5
    x = np.random.randn(batch, timesteps, features)
    
    layer = GRU(units, return_sequences=True)
    layer.build(x.shape)
    
    out = layer.forward(x)
    grad_out = np.random.randn(*out.shape)
    dx = layer.backward(grad_out)
    
    assert dx.shape == x.shape
    assert layer.grads['W'].shape == layer.params['W'].shape
    assert layer.grads['U'].shape == layer.params['U'].shape
    assert layer.grads['b'].shape == layer.params['b'].shape

def test_gru_sequential():
    model = Sequential([
        GRU(16, return_sequences=True, input_shape=(10, 8)),
        GRU(8)
    ])
    
    x = np.random.randn(2, 10, 8)
    out = model.forward(x)
    assert out.shape == (2, 8)
