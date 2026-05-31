import numpy as np
from neutro.layers.core.reparameterization import Reparameterization

def test_reparameterization():
    layer = Reparameterization()
    mean = np.zeros((10, 5))
    log_var = np.zeros((10, 5))
    
    # In training mode, it should be stochastic
    out1 = layer.forward([mean, log_var], training=True)
    out2 = layer.forward([mean, log_var], training=True)
    assert not np.array_equal(out1, out2)
    
    # In inference mode, it should return mean
    out_inf = layer.forward([mean, log_var], training=False)
    assert np.array_equal(out_inf, mean)
    
    # Test backward
    grad_output = np.ones((10, 5))
    # We need to run forward training once to set self.epsilon and self.z_log_var
    layer.forward([mean, log_var], training=True)
    grads = layer.backward(grad_output)
    assert len(grads) == 2
    assert grads[0].shape == (10, 5) # grad_mean
    assert grads[1].shape == (10, 5) # grad_log_var

def test_reparameterization_compute_output_shape():
    layer = Reparameterization()
    shape = layer.compute_output_shape([(10, 5), (10, 5)])
    assert shape == (10, 5)

def test_reparameterization_compute_output_shape_single():
    layer = Reparameterization()
    shape = layer.compute_output_shape((10, 5))
    assert shape == (10, 5)

def test_reparameterization_backward_shapes():
    layer = Reparameterization()
    mean = np.random.randn(4, 8)
    log_var = np.random.randn(4, 8)
    layer.forward([mean, log_var], training=True)

    grad_output = np.random.randn(4, 8)
    grads = layer.backward(grad_output)

    assert len(grads) == 2
    assert grads[0].shape == (4, 8)
    assert grads[1].shape == (4, 8)

def test_reparameterization_backward_values():
    layer = Reparameterization()
    mean = np.zeros((3, 2))
    log_var = np.ones((3, 2))  # var = exp(log_var) = e

    layer.forward([mean, log_var], training=True)
    grad_output = np.ones((3, 2))

    grads = layer.backward(grad_output)
    assert np.allclose(grads[0], np.ones((3, 2)))
    expected_log_var = np.ones((3, 2)) * np.exp(0.5) * 0.5 * layer.epsilon
    assert np.allclose(grads[1], expected_log_var)
