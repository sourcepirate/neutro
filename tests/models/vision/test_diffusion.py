import numpy as np
from neutro.models.vision.unet import UNet
from neutro.models.vision.diffusion_model import DiffusionModel

def test_diffusion_forward():
    unet = UNet(input_channels=1, base_filters=8)
    model = DiffusionModel(unet, timesteps=10)
    
    x = np.random.randn(2, 16, 16, 1)
    # Test training forward
    out = model.forward(x, training=True)
    assert out.shape == (2, 16, 16, 1)
    
    # Test sampling (very few steps for speed)
    sample = model.sample(shape=(1, 16, 16, 1))
    assert sample.shape == (1, 16, 16, 1)

def test_unet_backward():
    unet = UNet(input_channels=1, base_filters=4)
    x = np.random.randn(1, 8, 8, 1)
    t = np.array([5])
    
    # We need to run forward to populate layer caches
    out = unet.forward([x, t])
    grad = unet.backward(np.random.randn(1, 8, 8, 1))
    assert grad.shape == (1, 8, 8, 1)
