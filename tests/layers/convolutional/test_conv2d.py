import numpy as np
from neutro.layers.convolutional.conv2d import Conv2D

def test_conv2d_valid():
    layer = Conv2D(filters=4, kernel_size=3, padding='valid')
    x = np.random.rand(2, 10, 10, 1)
    out = layer(x)
    assert out.shape == (2, 8, 8, 4)
    
    grad = layer.backward(np.random.rand(2, 8, 8, 4))
    assert grad.shape == (2, 10, 10, 1)

def test_conv2d_same():
    layer = Conv2D(filters=4, kernel_size=3, padding='same')
    x = np.random.rand(2, 10, 10, 1)
    out = layer(x)
    assert out.shape == (2, 10, 10, 4)
    
    grad = layer.backward(np.random.rand(2, 10, 10, 4))
    assert grad.shape == (2, 10, 10, 1)

def test_conv2d_channels_first():
    layer = Conv2D(filters=4, kernel_size=3, padding='same', data_format='channels_first')
    x = np.random.rand(2, 1, 10, 10)
    out = layer(x)
    assert out.shape == (2, 4, 10, 10)

    grad = layer.backward(np.random.rand(2, 4, 10, 10))
    assert grad.shape == (2, 1, 10, 10)
