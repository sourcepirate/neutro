import numpy as np
import pytest
from neutro.activations.softmax import Softmax
from neutro.activations import get, ReLU

def test_softmax_forward():
    softmax = Softmax()
    x = np.array([[1, 2, 3]], dtype=float)
    out = softmax(x)
    assert np.allclose(np.sum(out), 1.0)
    assert out[0, 2] > out[0, 1] > out[0, 0]

def test_softmax_gradient_raises():
    # Softmax.gradient() is intentionally not implemented because s*(1-s) is
    # only the diagonal of the Jacobian and gives incorrect chain-rule results.
    softmax = Softmax()
    x = np.array([[1, 2, 3]], dtype=float)
    softmax(x)
    with pytest.raises(NotImplementedError):
        softmax.gradient(x)

def test_softmax_gradient_fast():
    # gradient_fast implements the correct vectorised backward:
    #   dL/dx = s * (g - dot(s, g))
    # which is equivalent to the full Jacobian J = diag(s) - s*s^T applied to g.
    softmax = Softmax()
    x = np.array([[1.0, 2.0, 3.0]], dtype=float)
    s = softmax(x)  # populates last_output

    grad_output = np.array([[1.0, 0.0, 0.0]], dtype=float)

    # Expected: s * (g - dot(s, g)) for each sample
    dot = np.sum(s * grad_output, axis=-1, keepdims=True)
    expected = s * (grad_output - dot)

    grad = softmax.gradient_fast(x, grad_output)
    assert grad.shape == x.shape
    assert np.allclose(grad, expected)

def test_softmax_gradient_fast_sum_zero():
    # A key property: the gradient of softmax sums to zero along the last axis
    # (because softmax output sums to 1, the Jacobian rows sum to 0).
    softmax = Softmax()
    x = np.array([[0.5, 1.5, -0.5, 2.0]], dtype=float)
    softmax(x)
    grad_output = np.random.default_rng(42).standard_normal(x.shape)
    grad = softmax.gradient_fast(x, grad_output)
    assert np.allclose(grad.sum(axis=-1), 0.0, atol=1e-12)

def test_get_activation():
    assert isinstance(get('relu'), ReLU)
    assert isinstance(get(ReLU()), ReLU)
    assert get(None) is None
