import numpy as np
from .base import Activation

class Softmax(Activation):
    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.last_output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.last_output
    def gradient(self, x):
        # Softmax does not have a valid element-wise gradient because its
        # Jacobian is a full matrix (diag(s) - s*s^T), not a diagonal one.
        # s*(1-s) is only the diagonal and produces incorrect gradients.
        # Always use gradient_fast(x, grad_output) instead.
        raise NotImplementedError(
            "Softmax.gradient() is not implemented because the softmax Jacobian "
            "is not diagonal. Use gradient_fast(x, grad_output) for correct "
            "chain-rule gradients: dL/dx = s * (g - dot(s, g))."
        )
    def gradient_fast(self, x, grad_output):
        # Vectorized softmax backward: dL/dx = s * (g - dot(s, g))
        # Derived from the full Jacobian J = diag(s) - s*s^T:
        # dL/dx_k = s_k * (g_k - sum_i(s_i * g_i))
        s = self.last_output
        dot = np.sum(s * grad_output, axis=-1, keepdims=True)
        return s * (grad_output - dot)
