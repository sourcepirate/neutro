import numpy as np
import pytest
from neutro.layers import BatchNormalization

def test_batch_norm_forward_training():
    batch, h, w, c = 2, 4, 4, 3
    layer = BatchNormalization()
    layer.build((batch, h, w, c))
    
    inputs = np.random.randn(batch, h, w, c) * 10 + 5 # mean 5, std 10
    out = layer.forward(inputs, training=True)
    
    assert out.shape == (batch, h, w, c)
    # Normalized output should have mean close to 0 and std close to 1 (if gamma=1, beta=0)
    assert np.allclose(np.mean(out, axis=(0, 1, 2)), 0, atol=1e-1)
    assert np.allclose(np.std(out, axis=(0, 1, 2)), 1, atol=1e-1)

def test_batch_norm_backward():
    batch, c = 4, 3
    layer = BatchNormalization()
    layer.build((batch, c))
    
    inputs = np.random.randn(batch, c)
    layer.forward(inputs, training=True)
    
    grad_output = np.random.randn(batch, c)
    grad_input = layer.backward(grad_output)
    
    assert grad_input.shape == (batch, c)
    assert layer.grads['gamma'].shape == (c,)
    assert layer.grads['beta'].shape == (c,)

def test_batch_norm_inference():
    batch, c = 4, 3
    layer = BatchNormalization(momentum=0.5)
    layer.build((batch, c))

    inputs = np.random.randn(batch, c) * 10 + 5
    out_train = layer.forward(inputs, training=True)
    assert out_train.shape == (batch, c)

    out_infer = layer.forward(inputs, training=False)
    assert out_infer.shape == (batch, c)
    assert np.allclose(layer.running_mean, 0.5 * 0 + 0.5 * np.mean(inputs, axis=0), atol=1e-6)
    assert np.allclose(layer.running_var, 0.5 * 1 + 0.5 * np.var(inputs, axis=0), atol=1e-6)
