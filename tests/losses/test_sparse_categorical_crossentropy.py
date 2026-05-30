import numpy as np
import pytest
from neutro.losses import SparseCategoricalCrossentropy, get


def test_scce_call_2d():
    scce = SparseCategoricalCrossentropy()
    y_true = np.array([0, 1])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
    loss = scce(y_true, y_pred)
    assert loss > 0


def test_scce_call_3d():
    scce = SparseCategoricalCrossentropy()
    # (batch=2, seq=3, vocab=4)
    y_true = np.array([[0, 1, 2], [3, 0, 1]])
    y_pred = np.random.dirichlet(np.ones(4), size=(2, 3))
    loss = scce(y_true, y_pred)
    assert loss > 0


def test_scce_gradient_2d():
    scce = SparseCategoricalCrossentropy()
    y_true = np.array([0, 1])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
    grad = scce.gradient(y_true, y_pred)
    assert grad.shape == (2, 2)
    # gradient should be non-zero only at true label positions
    assert grad[0, 0] < 0
    assert grad[0, 1] == 0.0
    assert grad[1, 1] < 0
    assert grad[1, 0] == 0.0


def test_scce_gradient_3d():
    scce = SparseCategoricalCrossentropy()
    y_true = np.array([[0, 1], [2, 3]])
    y_pred = np.random.dirichlet(np.ones(4), size=(2, 2))
    grad = scce.gradient(y_true, y_pred)
    assert grad.shape == (2, 2, 4)


def test_scce_shape_mismatch_raises():
    scce = SparseCategoricalCrossentropy()
    # y_true (batch,) vs y_pred (batch, seq, vocab) is a shape mismatch
    y_true = np.array([0, 1])
    y_pred = np.random.dirichlet(np.ones(4), size=(2, 3))
    with pytest.raises(ValueError):
        scce(y_true, y_pred)


def test_scce_get():
    loss = get('sparse_categorical_crossentropy')
    assert isinstance(loss, SparseCategoricalCrossentropy)
