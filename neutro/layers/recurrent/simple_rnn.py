import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer
from neutro.autograd import Tensor, ops as autograd_ops


class SimpleRNN(Layer):
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.activation_name = activation

    def build(self, input_shape):
        self.features = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['Wx'] = Tensor(init((self.features, self.units)))
        self.params['Wh'] = Tensor(init((self.units, self.units)))
        self.params['b'] = Tensor(get_initializer('zeros')((self.units,)))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)

    def forward(self, inputs, training=False):
        batch, timesteps, _ = inputs.shape
        u = self.units
        h = Tensor(np.zeros((batch, u)))

        hs = []
        for t in range(timesteps):
            z = inputs[:, t, :] @ self.params['Wx'] + h @ self.params['Wh'] + self.params['b']
            if self.activation_name == 'tanh':
                h = autograd_ops.tanh(z)
            else:
                h = z
            hs.append(h)

        if self.return_sequences:
            return autograd_ops.concatenate([h_t.reshape(batch, 1, u) for h_t in hs], axis=1)
        return h