import numpy as np
from .base import Activation

class Softmax(Activation):
    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.last_output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.last_output
    def gradient(self, x):
        return self.last_output * (1 - self.last_output)
    def gradient_fast(self, x, grad_output):
        orig_shape = grad_output.shape
        grad_flat = grad_output.reshape(-1, orig_shape[-1])
        out_flat = self.last_output.reshape(-1, orig_shape[-1])
        
        n_samples, units = grad_flat.shape
        res = np.zeros_like(grad_flat)
        for i in range(n_samples):
            s = out_flat[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            res[i] = np.dot(grad_flat[i], jacobian)
        return res.reshape(orig_shape)
