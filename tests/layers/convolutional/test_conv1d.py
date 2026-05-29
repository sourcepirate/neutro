import numpy as np
import pytest
from neutro.layers import Conv1D

def test_conv1d_forward_valid():
    batch, steps, channels = 2, 10, 3
    filters, kernel_size = 4, 3
    layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid')
    layer.build((batch, steps, channels))
    
    inputs = np.random.randn(batch, steps, channels)
    out = layer.forward(inputs)
    
    # (10 - 3) // 1 + 1 = 8
    assert out.shape == (batch, 8, filters)

def test_conv1d_forward_same():
    batch, steps, channels = 2, 10, 3
    filters, kernel_size = 4, 3
    layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')
    layer.build((batch, steps, channels))
    
    inputs = np.random.randn(batch, steps, channels)
    out = layer.forward(inputs)
    
    assert out.shape == (batch, 10, filters)

def test_conv1d_backward():
    batch, steps, channels = 2, 10, 3
    filters, kernel_size = 4, 3
    layer = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid')
    layer.build((batch, steps, channels))
    
    inputs = np.random.randn(batch, steps, channels)
    layer.forward(inputs)
    
    grad_output = np.random.randn(batch, 8, filters)
    grad_input = layer.backward(grad_output)
    
    assert grad_input.shape == inputs.shape
    assert layer.grads['W'].shape == layer.params['W'].shape
    assert layer.grads['b'].shape == layer.params['b'].shape
