import numpy as np
from neutro.metrics import Precision, Recall, F1Score

def test_precision():
    p = Precision()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    # TP=1, FP=1 -> P = 1/2 = 0.5
    assert np.isclose(p(y_true, y_pred), 0.5)

def test_recall():
    r = Recall()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    # TP=1, FN=1 -> R = 1/2 = 0.5
    assert np.isclose(r(y_true, y_pred), 0.5)

def test_f1_score():
    f1 = F1Score()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    # P=0.5, R=0.5 -> F1 = 0.5
    assert np.isclose(f1(y_true, y_pred), 0.5)
