import numpy as np
from neutro.models.vision.vae import VAE

def test_vae_shapes():
    input_shape = (28, 28, 1)
    latent_dim = 16
    model = VAE(input_shape, latent_dim)
    
    x = np.random.randn(2, 28, 28, 1)
    out = model(x, training=True)
    assert out.shape == (2, 28, 28, 1)
    
    grad = model.backward(np.random.randn(2, 28, 28, 1))
    assert grad.shape == (2, 28, 28, 1)

def test_vae_inference():
    input_shape = (28, 28, 1)
    latent_dim = 16
    model = VAE(input_shape, latent_dim)
    
    x = np.random.randn(1, 28, 28, 1)
    # In inference mode, forward should be deterministic
    out1 = model(x, training=False)
    out2 = model(x, training=False)
    assert np.allclose(out1, out2)
