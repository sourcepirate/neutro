import numpy as np
from .base import Loss

class VAELoss(Loss):
    """
    VAE Loss: Reconstruction Loss + KL Divergence.
    
    Args:
        model: The VAE model instance containing z_mean and z_log_var.
        reconstruction_loss_type: 'mse' or 'bce'.
        kl_weight: Weight for the KL divergence term.
    """
    def __init__(self, model, reconstruction_loss_type='mse', kl_weight=1.0):
        self.model = model
        self.reconstruction_loss_type = reconstruction_loss_type
        self.kl_weight = kl_weight

    def __call__(self, y_true, y_pred):
        # Reconstruction loss
        if self.reconstruction_loss_type == 'mse':
            recon_loss = np.mean(np.square(y_true - y_pred))
        else:
            # Simple BCE implementation
            epsilon = 1e-12
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            recon_loss = -np.mean(y_true * np.log(y_pred) + (1. - y_true) * np.log(1. - y_pred))
            
        # KL Divergence: 0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        z_mean = self.model.z_mean
        z_log_var = self.model.z_log_var
        kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
        
        return recon_loss + self.kl_weight * kl_loss

    def gradient(self, y_true, y_pred):
        """
        Gradient of the reconstruction loss part. 
        The KL divergence gradient is handled separately in the VAE model's backward 
        if we want to be exact, but for simplicity here we return recon gradient 
        and assume the bottleneck backward handles KL.
        
        Actually, a cleaner way is for the bottleneck (sampling layer) 
        to add the KL gradient to its own grads during its backward.
        """
        if self.reconstruction_loss_type == 'mse':
            return -2 * (y_true - y_pred) / y_true.size
        else:
            epsilon = 1e-12
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            return (y_pred - y_true) / (y_pred * (1.0 - y_pred)) / y_true.size
