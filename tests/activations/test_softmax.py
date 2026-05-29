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

def test_softmax_gradient():
    softmax = Softmax()
    x = np.array([[1, 2, 3]], dtype=float)
    softmax(x)
    grad = softmax.gradient(x)
    assert grad.shape == x.shape

def test_softmax_gradient_fast():
    softmax = Softmax()
    x = np.array([[1, 2]], dtype=float)
    out = softmax(x)
    grad_output = np.array([[1, 0]], dtype=float)
    grad = softmax.gradient_fast(x, grad_output)
    assert grad.shape == x.shape

def test_get_activation():
    assert isinstance(get('relu'), ReLU)
    assert isinstance(get(ReLU()), ReLU)
    assert get(None) is None
