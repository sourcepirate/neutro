import numpy as np
from neutro.layers.transformer.transformer_block import TransformerBlock

def test_transformer_block():
    layer = TransformerBlock(embed_dim=16, num_heads=2, ff_dim=32)
    x = np.random.rand(2, 5, 16)
    out = layer(x)
    assert out.shape == (2, 5, 16)
    
    grad = layer.backward(np.random.rand(2, 5, 16))
    assert grad.shape == (2, 5, 16)
