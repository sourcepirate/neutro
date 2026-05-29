import numpy as np
from neutro.optimizers import SGD, Adam, AdamW
from neutro.layers import Dense

class MockLayer:
    def __init__(self):
        self.params = {'W': np.array([1.0, 2.0]), 'b': np.array([0.0])}
        self.grads = {'W': np.array([0.1, 0.1]), 'b': np.array([0.01])}
        self.trainable = True

def test_sgd():
    layer = MockLayer()
    opt = SGD(learning_rate=0.1, momentum=0.9)
    opt.step([layer])
    assert not np.all(layer.params['W'] == np.array([1.0, 2.0]))

def test_sgd_nesterov():
    layer = MockLayer()
    opt = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    opt.step([layer])
    assert not np.all(layer.params['W'] == np.array([1.0, 2.0]))

def test_adam():
    layer = MockLayer()
    opt = Adam(learning_rate=0.1)
    opt.step([layer])
    assert not np.all(layer.params['W'] == np.array([1.0, 2.0]))

def test_adamw():
    layer = MockLayer()
    opt = AdamW(learning_rate=0.1, weight_decay=0.01)
    opt.step([layer])
    assert not np.all(layer.params['W'] == np.array([1.0, 2.0]))
