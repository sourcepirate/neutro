import numpy as np
import pytest
from neutro.activations.relu import ReLU

def test_relu_forward():
    relu = ReLU()
    x = np.array([-1, 0, 1], dtype=float)
    expected = np.array([0, 0, 1], dtype=float)
    assert np.all(relu(x) == expected)

def test_relu_gradient():
    relu = ReLU()
    x = np.array([-1, 0, 1], dtype=float)
    expected = np.array([0, 0, 1], dtype=float)
    assert np.all(relu.gradient(x) == expected)
