import numpy as np
from ..base import Layer
from neutro.autograd import Tensor
from neutro.autograd.function import Function


class DropoutFunction(Function):
    """
    Inverted dropout:
        y = x * mask / (1 - rate),   mask ~ Bernoulli(1 - rate)

    During backward the same mask (stored in ctx) is reused.
    """
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask=mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_data['mask']
        return grad_output * mask, None


class Dropout(Layer):
    """Inverted dropout layer."""

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def forward(self, inputs, training=False):
        if not training or self.rate == 0:
            self.mask = None
            return inputs
        mask_np = np.random.binomial(
            1, 1 - self.rate, size=inputs.shape,
        ).astype(float) / (1 - self.rate)
        self.mask = mask_np
        return DropoutFunction.apply(inputs, Tensor(mask_np))

    def compute_output_shape(self, input_shape):
        return input_shape