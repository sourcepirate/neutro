import numpy as np
from neutro.layers.recurrent.simple_rnn import SimpleRNN

def test_simple_rnn():
    layer = SimpleRNN(units=16, return_sequences=True)
    x = np.random.rand(2, 5, 8)
    out = layer(x)
    assert out.shape == (2, 5, 16)
    
    grad = layer.backward(np.random.rand(2, 5, 16))
    assert grad.shape == (2, 5, 8)

def test_simple_rnn_no_seq():
    layer = SimpleRNN(units=16, return_sequences=False)
    x = np.random.rand(2, 5, 8)
    out = layer(x)
    assert out.shape == (2, 16)
    
    grad = layer.backward(np.random.rand(2, 16))
    assert grad.shape == (2, 5, 8)
