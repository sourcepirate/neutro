import numpy as np
from neutro.layers.core.reparameterization import Reparameterization


def test_reparameterization():
    layer = Reparameterization()
    mean = np.zeros((10, 5))
    log_var = np.zeros((10, 5))

    out1 = layer([mean, log_var], training=True)
    out2 = layer([mean, log_var], training=True)
    assert not np.array_equal(out1, out2)

    out_inf = layer([mean, log_var], training=False)
    assert np.array_equal(out_inf, mean)

    grad_output = np.ones((10, 5))
    layer([mean, log_var], training=True)
    grads = layer.backward(grad_output)
    assert len(grads) == 2
    assert grads[0].shape == (10, 5)
    assert grads[1].shape == (10, 5)


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
    layer([mean, log_var], training=True)

    grad_output = np.random.randn(4, 8)
    grads = layer.backward(grad_output)

    assert len(grads) == 2
    assert grads[0].shape == (4, 8)
    assert grads[1].shape == (4, 8)


def test_reparameterization_backward_values():
    layer = Reparameterization()
    mean = np.zeros((3, 2))
    log_var = np.ones((3, 2))

    layer([mean, log_var], training=True)
    grad_output = np.ones((3, 2))

    grads = layer.backward(grad_output)
    assert np.allclose(grads[0], np.ones((3, 2)))
    expected_log_var = np.ones((3, 2)) * np.exp(0.5) * 0.5 * layer.epsilon
    assert np.allclose(grads[1], expected_log_var)
