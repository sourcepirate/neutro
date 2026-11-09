import numpy as np


class DiscreteDiffusionScheduler:
    """Absorbing-mask scheduler for masked diffusion LMs (LLaDA/MDLM-style).

    Forward: each token masked independently with prob sigma(t).
    sigma(t) = 1 - alpha_bar(t); linear or cosine schedule.
    """

    def __init__(self, timesteps=1000, schedule='linear', mask_id=0):
        self.timesteps = timesteps
        self.schedule = schedule
        self.mask_id = mask_id
        steps = np.arange(timesteps + 1) / timesteps  # 0..1
        if schedule == 'cosine':
            self.alpha_bar = np.cos(steps * np.pi / 2) ** 2
        else:
            self.alpha_bar = 1.0 - steps
        self.alpha_bar = np.clip(self.alpha_bar, 1e-6, 1.0)

    def mask_prob(self, t):
        t = np.asarray(t, dtype=np.int64).clip(0, self.timesteps)
        return 1.0 - self.alpha_bar[t]

    def q_sample(self, x0, t, rng=None):
        """Mask x0 with per-token prob sigma(t). Returns (xt, mask)."""
        rng = rng or np.random.default_rng()
        x0 = np.asarray(x0)
        if np.ndim(t) == 0:
            p = float(self.mask_prob(t))
            probs = np.full(x0.shape, p)
        else:
            p = self.mask_prob(np.asarray(t)).reshape(-1, *([1] * (x0.ndim - 1)))
            probs = np.broadcast_to(p, x0.shape)
        mask = rng.random(x0.shape) < probs
        xt = x0.copy()
        xt[mask] = self.mask_id
        return xt, mask
