import numpy as np
from ..base_model import Model
from ...utils.diffusion_utils import GaussianDiffusion

class DiffusionModel(Model):
    """
    Wrapper for Diffusion training and sampling.
    """
    def __init__(self, unet, timesteps=1000):
        super().__init__()
        self.unet = unet
        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        self.layers = self.unet.layers

    def forward(self, x_start, training=False):
        if training:
            batch_size = x_start.shape[0]
            t = np.random.randint(0, self.diffusion.timesteps, (batch_size,))
            noise = np.random.normal(size=x_start.shape)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)
            predicted_noise = self.unet([x_t, t], training=True)
            # Store for loss calculation
            self.current_noise = noise
            self.predicted_noise = predicted_noise
            return predicted_noise
        else:
            # Inference mode: just pass through UNet
            return self.unet(x_start, training=False)

    def backward(self, grad):
        # In diffusion training, grad usually comes from (predicted_noise - actual_noise)
        return self.unet.backward(grad)

    def sample(self, shape):
        """
        Sample from the model (denoising process).
        """
        batch_size = shape[0]
        x = np.random.normal(size=shape)
        
        for t in reversed(range(self.diffusion.timesteps)):
            x = self.diffusion.p_sample(self.unet, x, t)
            
        return x
