import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer
from neutro.autograd import Tensor, ops as autograd_ops


class GRU(Layer):
    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.features = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['W'] = Tensor(init((self.features, 3 * self.units)))
        self.params['U'] = Tensor(init((self.units, 3 * self.units)))
        self.params['b'] = Tensor(get_initializer('zeros')((3 * self.units,)))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)

    def forward(self, inputs, training=False):
        batch, timesteps, _ = inputs.shape
        u = self.units
        h = Tensor(np.zeros((batch, u)))

        x_W = inputs @ self.params['W'] + self.params['b']

        hs = []
        for t in range(timesteps):
            x_W_t = x_W[:, t, :]
            h_prev = h

            z_r_logits = x_W_t[:, :2*u] + h_prev @ self.params['U'][:, :2*u]
            z = autograd_ops.sigmoid(z_r_logits[:, :u])
            r = autograd_ops.sigmoid(z_r_logits[:, u:])

            h_tilde_logits = x_W_t[:, 2*u:] + (r * h_prev) @ self.params['U'][:, 2*u:]
            h_tilde = autograd_ops.tanh(h_tilde_logits)

            h = (1.0 - z) * h_tilde + z * h_prev
            hs.append(h)

        if self.return_sequences:
            return autograd_ops.concatenate([h_t.reshape(batch, 1, u) for h_t in hs], axis=1)
        return h