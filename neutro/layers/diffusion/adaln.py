from ..base import Layer
from ..core.dense import Dense
from ..normalization.layernorm import LayerNormalization


class AdaLN(Layer):
    """Adaptive LayerNorm: scale/shift from conditioning vector.

    y = LN(x) * (1 + scale) + shift, with scale/shift = Dense(c) zero-init.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.norm = LayerNormalization(epsilon=epsilon)
        self.scale_proj = Dense(0)  # units set in build
        self.shift_proj = Dense(0)

    def build(self, input_shape):
        # input_shape: [x_shape, c_shape]
        x_shape, c_shape = input_shape
        dim = x_shape[-1]
        self.scale_proj.units = dim
        self.shift_proj.units = dim
        self.norm.build(x_shape)
        self.scale_proj.build(c_shape)
        self.shift_proj.build(c_shape)
        # Zero-init modulation so block starts as identity.
        import numpy as np
        self.scale_proj.params['W'].data[:] = 0.0
        self.shift_proj.params['W'].data[:] = 0.0
        if 'b' in self.scale_proj.params:
            self.scale_proj.params['b'].data[:] = 0.0
        if 'b' in self.shift_proj.params:
            self.shift_proj.params['b'].data[:] = 0.0
        super().build(x_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def forward(self, inputs, training=False):
        x, cond = inputs
        h = self.norm(x, training)
        # cond: (B, C) -> broadcast over sequence: (B, 1, D)
        scale = self.scale_proj(cond, training)
        shift = self.shift_proj(cond, training)
        while scale.ndim < h.ndim:
            scale = scale.reshape(scale.shape[:-1] + (1,) + scale.shape[-1:])
            shift = shift.reshape(shift.shape[:-1] + (1,) + shift.shape[-1:])
        return h * (1 + scale) + shift


class AdaLNZero(AdaLN):
    """AdaLN with zero-initialized gate on residual branch."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)
        self.gate_proj = Dense(0)

    def build(self, input_shape):
        x_shape, c_shape = input_shape
        super().build(input_shape)
        dim = x_shape[-1]
        self.gate_proj.units = dim
        self.gate_proj.build(c_shape)
        import numpy as np
        self.gate_proj.params['W'].data[:] = 0.0
        if 'b' in self.gate_proj.params:
            self.gate_proj.params['b'].data[:] = 0.0

    def modulate(self, x, branch_out, cond, training=False):
        gate = self.gate_proj(cond, training)
        while gate.ndim < branch_out.ndim:
            gate = gate.reshape(gate.shape[:-1] + (1,) + gate.shape[-1:])
        normed = self.forward([x, cond], training)
        return normed, gate * branch_out
