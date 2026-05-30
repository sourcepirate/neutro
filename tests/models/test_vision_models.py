import numpy as np
from neutro.models.vision.vgg import VGG16, VGG19
from neutro.models.vision.alexnet import AlexNet

def test_vgg16_smoke():
    # Use smaller input for smoke test
    model = VGG16(input_shape=(32, 32, 3), num_classes=10)
    x = np.random.randn(1, 32, 32, 3)
    y = model.predict(x)
    assert y.shape == (1, 10)
    assert np.allclose(np.sum(y), 1.0)

def test_vgg19_smoke():
    model = VGG19(input_shape=(32, 32, 3), num_classes=10)
    x = np.random.randn(1, 32, 32, 3)
    y = model.predict(x)
    assert y.shape == (1, 10)
    assert np.allclose(np.sum(y), 1.0)

def test_alexnet_smoke():
    # AlexNet has large kernels and strides, so input must be large enough
    model = AlexNet(input_shape=(227, 227, 3), num_classes=10)
    x = np.random.randn(1, 227, 227, 3)
    y = model.predict(x)
    assert y.shape == (1, 10)
    assert np.allclose(np.sum(y), 1.0)

def test_vgg16_channels_first_smoke():
    model = VGG16(input_shape=(3, 32, 32), num_classes=10, data_format='channels_first')
    x = np.random.randn(1, 3, 32, 32)
    y = model.predict(x)
    assert y.shape == (1, 10)
    assert np.allclose(np.sum(y), 1.0)
