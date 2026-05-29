import numpy as np
import pytest
from neutro.activations.sigmoid import Sigmoid

def test_sigmoid_forward():
    sigmoid = Sigmoid()
    x = np.array([0], dtype=float)
    assert sigmoid(x) == 0.5

def test_sigmoid_gradient():
    sigmoid = Sigmoid()
    x = np.array([0], dtype=float)
    # sigmoid(0) = 0.5, grad = 0.5 * (1 - 0.5) = 0.25
    assert sigmoid.gradient(x) == 0.25
