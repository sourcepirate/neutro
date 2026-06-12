import numpy as np
from neutro.layers.core.bitlinear import BitLinear


def test_bitlinear_b1_forward():
    layer = BitLinear(10, mode='b1', activation_bits=8)
    x = np.random.randn(5, 4)
    out = layer(x)
    assert out.shape == (5, 10)


def test_bitlinear_b158_forward():
    layer = BitLinear(10, mode='b1.58', activation_bits=8)
    x = np.random.randn(5, 4)
    out = layer(x)
    assert out.shape == (5, 10)


def test_bitlinear_backward():
    layer = BitLinear(10, mode='b1.58')
    x = np.random.randn(5, 4)
    out = layer(x)
    grad = layer.backward(np.random.randn(5, 10))
    assert grad.shape == (5, 4)
    assert 'W' in layer.grads
    assert layer.grads['W'].shape == (4, 10)


def test_bitlinear_b1_backward():
    layer = BitLinear(10, mode='b1')
    x = np.random.randn(5, 4)
    out = layer(x)
    grad = layer.backward(np.random.randn(5, 10))
    assert grad.shape == (5, 4)
    assert 'W' in layer.grads


def test_bitlinear_gamma_beta_params():
    layer = BitLinear(10)
    x = np.random.randn(5, 4)
    layer(x)
    assert 'gamma_ln' in layer.params
    assert 'beta_ln' in layer.params
    assert layer.params['gamma_ln'].shape == (4,)
    assert layer.params['beta_ln'].shape == (4,)


def test_bitlinear_gamma_beta_grads():
    layer = BitLinear(10)
    x = np.random.randn(5, 4)
    layer(x)
    _ = layer.backward(np.random.randn(5, 10))
    assert 'gamma_ln' in layer.grads
    assert 'beta_ln' in layer.grads


def test_bitlinear_activation_quantization():
    x = np.random.randn(3, 8)
    from neutro.layers.core.bitlinear import activation_quantize
    x_q, gamma = activation_quantize(x, bits=8, per_token=False)
    assert x_q.shape == x.shape
    assert np.all(x_q >= -127)
    assert np.all(x_q <= 127)
    assert gamma > 0


def test_bitlinear_weight_quantize_b1():
    from neutro.layers.core.bitlinear import weight_quantize_b1
    W = np.random.randn(8, 16)
    W_q, beta = weight_quantize_b1(W)
    assert np.all(np.abs(W_q) == 1.0)
    assert beta > 0


def test_bitlinear_weight_quantize_b158():
    from neutro.layers.core.bitlinear import weight_quantize_b158
    W = np.random.randn(8, 16)
    W_q, beta = weight_quantize_b158(W)
    assert np.all(np.abs(W_q) <= 1.0 + 1e-10)
    assert beta > 0
