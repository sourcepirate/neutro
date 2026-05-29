import numpy as np
import pytest
from neutro.losses import MeanSquaredError, get

def test_mse_call():
    mse = MeanSquaredError()
    y_true = np.array([1, 2], dtype=float)
    y_pred = np.array([0, 2], dtype=float)
    assert mse(y_true, y_pred) == 0.5

def test_mse_gradient():
    mse = MeanSquaredError()
    y_true = np.array([1], dtype=float)
    y_pred = np.array([0], dtype=float)
    # grad = -2 * (1 - 0) / 1 = -2
    assert mse.gradient(y_true, y_pred) == -2.0

def test_get_loss():
    assert isinstance(get('mse'), MeanSquaredError)
