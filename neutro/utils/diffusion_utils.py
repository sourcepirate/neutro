import numpy as np

class GaussianDiffusion:
    """
    DDPM (Denoising Diffusion Probabilistic Models) scheduler.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = np.random.normal(size=x_start.shape)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x_t, t):
        """
        Reverse diffusion: sample x_{t-1} from p(x_{t-1} | x_t)
        """
        # Predict noise
        t_batch = np.full((x_t.shape[0],), t)
        predicted_noise = model.predict([x_t, t_batch])
        
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Mean of p(x_{t-1} | x_t)
        mu = (1 / np.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
        
        if t == 0:
            return mu
        else:
            noise = np.random.normal(size=x_t.shape)
            variance = self.betas[t] # Or use (1 - alpha_bar_{t-1})/(1 - alpha_bar_t) * beta_t
            return mu + np.sqrt(variance) * noise
