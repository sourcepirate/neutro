import numpy as np
from neutro.layers.core.dropout import Dropout

def test_dropout():
    layer = Dropout(0.5)
    x = np.random.rand(10, 10)
    
    # Inference
    out_inf = layer.forward(x, training=False)
    assert np.all(out_inf == x)
    
    # Training
    out_train = layer.forward(x, training=True)
    assert not np.all(out_train == x)
    
    grad = layer.backward(np.random.rand(10, 10))
    assert grad.shape == (10, 10)
