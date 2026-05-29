import numpy as np
import pytest
from neutro.losses import CategoricalCrossentropy

def test_cce_call():
    cce = CategoricalCrossentropy()
    y_true = np.array([[1, 0]], dtype=float)
    y_pred = np.array([[0.9, 0.1]], dtype=float)
    loss = cce(y_true, y_pred)
    assert loss > 0

def test_cce_gradient():
    cce = CategoricalCrossentropy()
    y_true = np.array([[1, 0]], dtype=float)
    y_pred = np.array([[0.5, 0.5]], dtype=float)
    grad = cce.gradient(y_true, y_pred)
    assert grad.shape == (1, 2)
