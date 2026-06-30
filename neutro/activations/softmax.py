import numpy as np
from .base import Activation
from neutro.autograd import ops as autograd_ops


class Softmax(Activation):
    def __call__(self, x):
        return autograd_ops.softmax(x)

    def gradient(self, x):
        raise NotImplementedError(
            "Softmax.gradient() is not implemented because the softmax Jacobian "
            "is not diagonal. Use gradient_fast(x, grad_output) for correct "
            "chain-rule gradients: dL/dx = s * (g - dot(s, g))."
        )

    def gradient_fast(self, x, grad_output):
        s = self(x).data if hasattr(self(x), 'data') else self(x)
        dot = np.sum(s * grad_output, axis=-1, keepdims=True)
        return s * (grad_output - dot)