import numpy as np
import pytest
from neutro.metrics import SparseAccuracy, get


def test_sparse_accuracy_2d():
    acc = SparseAccuracy()
    # y_true: (N,), y_pred: (N, C)
    y_true = np.array([0, 1, 2])
    y_pred = np.array([
        [0.9, 0.05, 0.05],  # correctly predicts class 0
        [0.1, 0.8,  0.1 ],  # correctly predicts class 1
        [0.3, 0.3,  0.4 ],  # correctly predicts class 2
    ])
    result = acc(y_true, y_pred)
    assert result == 1.0


def test_sparse_accuracy_2d_partial():
    acc = SparseAccuracy()
    y_true = np.array([0, 1])
    y_pred = np.array([
        [0.9, 0.1],  # correct
        [0.8, 0.2],  # wrong (predicts 0, true is 1)
    ])
    result = acc(y_true, y_pred)
    assert result == 0.5


def test_sparse_accuracy_3d():
    acc = SparseAccuracy()
    # y_true: (N, S), y_pred: (N, S, C)
    y_true = np.array([[0, 1], [2, 0]])
    y_pred = np.array([
        [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]],  # both correct
        [[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]],  # both correct
    ])
    result = acc(y_true, y_pred)
    assert result == 1.0


def test_sparse_accuracy_get_name():
    acc = SparseAccuracy()
    assert acc.get_name() == "sparse_accuracy"


def test_sparse_accuracy_get():
    metric = get('sparse_accuracy')
    assert isinstance(metric, SparseAccuracy)
