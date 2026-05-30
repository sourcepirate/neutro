import numpy as np
from ..base_model import Model
from ...layers.core.dense import Dense
from ...layers.core.reparameterization import Reparameterization
from ...layers.core.flatten import Flatten
from ...layers.convolutional.conv2d import Conv2D
from ...losses.vae_loss import VAELoss

class VAE(Model):
    """
    Variational Autoencoder.
    """
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.flatten = Flatten()
        self.fc_mean = Dense(latent_dim)
        self.fc_log_var = Dense(latent_dim)
        self.sampling = Reparameterization()
        
        # Decoder
        self.decoder_fc = Dense(np.prod(input_shape), activation='sigmoid') # Use sigmoid for BCE
        
        self.layers = [
            self.encoder_conv, self.flatten, self.fc_mean, self.fc_log_var, self.sampling,
            self.decoder_fc
        ]

    def forward(self, x, training=False):
        # Encoder
        h = self.encoder_conv(x, training)
        h = self.flatten(h)
        self.z_mean = self.fc_mean(h, training)
        self.z_log_var = self.fc_log_var(h, training)
        
        # Bottleneck
        z = self.sampling([self.z_mean, self.z_log_var], training)
        
        # Decoder
        reconstruction = self.decoder_fc(z, training)
        # Reshape reconstruction to input_shape
        return reconstruction.reshape(-1, *x.shape[1:])

    def backward(self, grad_output):
        # Backprop through decoder
        grad_z = self.decoder_fc.backward(grad_output.reshape(grad_output.shape[0], -1))
        
        # Add KL gradients if loss is VAELoss
        kl_weight = 1.0
        if hasattr(self, 'loss_fn') and isinstance(self.loss_fn, VAELoss):
            kl_weight = self.loss_fn.kl_weight
            
        N = self.z_mean.size
        dKL_dmean = kl_weight * self.z_mean / N
        dKL_dlogvar = kl_weight * 0.5 * (np.exp(self.z_log_var) - 1) / N
        
        # Backprop through sampling
        grad_mean_recon, grad_log_var_recon = self.sampling.backward(grad_z)
        
        grad_mean = grad_mean_recon + dKL_dmean
        grad_log_var = grad_log_var_recon + dKL_dlogvar
        
        # Backprop through encoder
        grad_h_mean = self.fc_mean.backward(grad_mean)
        grad_h_log_var = self.fc_log_var.backward(grad_log_var)
        grad_h = grad_h_mean + grad_h_log_var
        
        grad_conv = self.flatten.backward(grad_h)
        return self.encoder_conv.backward(grad_conv)
