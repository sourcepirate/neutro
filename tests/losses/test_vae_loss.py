import numpy as np
from neutro.losses.vae_loss import VAELoss

class MockModel:
    def __init__(self, mean, log_var):
        self.z_mean = mean
        self.z_log_var = log_var

def test_vae_loss():
    mean = np.zeros((1, 10))
    log_var = np.zeros((1, 10))
    model = MockModel(mean, log_var)
    
    loss_fn = VAELoss(model, reconstruction_loss_type='mse', kl_weight=1.0)
    
    y_true = np.ones((1, 28, 28, 1))
    y_pred = np.zeros((1, 28, 28, 1))
    
    # KL for N(0, 1) should be 0
    # Recon loss for all 1s vs all 0s should be 1.0
    loss = loss_fn(y_true, y_pred)
    assert np.allclose(loss, 1.0)
    
    # Gradient test
    grad = loss_fn.gradient(y_true, y_pred)
    assert grad.shape == y_true.shape
    assert np.all(grad < 0) # y_true > y_pred, so grad should be negative for MSE
