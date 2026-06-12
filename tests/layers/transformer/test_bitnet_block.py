import numpy as np
from neutro.layers.transformer.bitnet_block import BitNetBlock


def test_bitnet_block_forward():
    block = BitNetBlock(embed_dim=32, num_heads=4, ff_dim=64, mode='b1.58')
    x = np.random.randn(2, 8, 32)
    out = block(x)
    assert out.shape == (2, 8, 32)


def test_bitnet_block_b1_forward():
    block = BitNetBlock(embed_dim=32, num_heads=4, ff_dim=64, mode='b1')
    x = np.random.randn(2, 8, 32)
    out = block(x)
    assert out.shape == (2, 8, 32)


def test_bitnet_block_backward():
    block = BitNetBlock(embed_dim=32, num_heads=4, ff_dim=64, mode='b1.58')
    x = np.random.randn(2, 8, 32)
    out = block(x)
    grad = block.backward(np.random.randn(2, 8, 32))
    assert grad.shape == (2, 8, 32)


def test_bitnet_block_sublayers():
    block = BitNetBlock(embed_dim=32, num_heads=4, ff_dim=64, mode='b1.58')
    x = np.random.randn(2, 8, 32)
    block(x)
    layers = block.sublayers
    assert len(layers) > 0
