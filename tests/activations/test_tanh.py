import numpy as np
import pytest
from neutro.activations.tanh import Tanh

def test_tanh_forward():
    tanh = Tanh()
    x = np.array([0], dtype=float)
    assert tanh(x) == 0

def test_tanh_gradient():
    tanh = Tanh()
    x = np.array([0], dtype=float)
    # tanh(0) = 0, grad = 1 - 0^2 = 1
    assert tanh.gradient(x) == 1
