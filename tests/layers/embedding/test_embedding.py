import numpy as np
from neutro.layers.embedding.embedding import Embedding

def test_embedding():
    layer = Embedding(100, 16)
    x = np.random.randint(0, 100, (2, 5))
    out = layer(x)
    assert out.shape == (2, 5, 16)
    
    grad = layer.backward(np.random.rand(2, 5, 16))
    assert grad is None
    assert layer.grads['embeddings'].shape == (100, 16)
