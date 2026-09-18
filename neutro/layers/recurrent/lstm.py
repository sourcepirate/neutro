import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer
from neutro.autograd import Tensor, ops as autograd_ops


class LSTM(Layer):
    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.features = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['W'] = Tensor(init((self.features + self.units, 4 * self.units)))
        self.params['b'] = Tensor(get_initializer('zeros')((4 * self.units,)))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)

    def forward(self, inputs, training=False):
        batch, timesteps, _ = inputs.shape
        u = self.units
        h = Tensor(np.zeros((batch, u)))
        c = Tensor(np.zeros((batch, u)))

        hs = []
        for t in range(timesteps):
            x_t = inputs[:, t, :]
            concat = autograd_ops.concatenate([x_t, h], axis=1)
            z = concat @ self.params['W'] + self.params['b']
            i = autograd_ops.sigmoid(z[:, :u])
            f = autograd_ops.sigmoid(z[:, u:2*u])
            c_tilde = autograd_ops.tanh(z[:, 2*u:3*u])
            o = autograd_ops.sigmoid(z[:, 3*u:])
            c = f * c + i * c_tilde
            h = o * autograd_ops.tanh(c)
            hs.append(h)

        if self.return_sequences:
            return autograd_ops.concatenate([h_t.reshape(batch, 1, u) for h_t in hs], axis=1)
        return h