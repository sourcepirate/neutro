import numpy as np
from neutro.metrics import Accuracy, get

def test_accuracy():
    acc = Accuracy()
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])
    assert acc(y_true, y_pred) == 1.0
    
    y_true_cat = np.array([[1, 0], [0, 1]])
    y_pred_cat = np.array([[0.8, 0.2], [0.7, 0.3]])
    assert acc(y_true_cat, y_pred_cat) == 0.5

def test_get_metric():
    assert isinstance(get('accuracy'), Accuracy)
    assert get(None) is None
