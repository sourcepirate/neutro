import numpy as np
from .base import Activation

class SiLU(Activation):
    """
    SiLU (Sigmoid Linear Unit) or Swish activation function: x * sigmoid(x).
    Commonly used in Llama, Qwen, and DeepSeek.
    """
    def __call__(self, x):
        self.sigmoid_x = 1 / (1 + np.exp(-x))
        self.x = x
        return x * self.sigmoid_x

    def gradient(self, x):
        # f'(x) = f(x) + sigmoid(x) * (1 - f(x))
        f_x = x * self.sigmoid_x
        return f_x + self.sigmoid_x * (1 - f_x)

    def gradient_fast(self, x, grad_output):
        return grad_output * self.gradient(x)
