import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer

class LSTM(Layer):
    def __init__(self, units, return_sequences=False):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.features = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['W'] = init((self.features + self.units, 4 * self.units))
        self.params['b'] = get_initializer('zeros')((4 * self.units,))
        super().build(input_shape)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, inputs, training=False):
        self.inputs = inputs
        batch, timesteps, _ = inputs.shape
        self.h_states = np.zeros((batch, timesteps + 1, self.units))
        self.c_states = np.zeros((batch, timesteps + 1, self.units))
        self.gates = np.zeros((batch, timesteps, 4 * self.units))
        
        for t in range(timesteps):
            concat = np.concatenate([inputs[:, t, :], self.h_states[:, t, :]], axis=1)
            z = np.dot(concat, self.params['W']) + self.params['b']
            self.gates[:, t, :] = z
            i, f, c_tilde, o = self._sigmoid(z[:, :self.units]), self._sigmoid(z[:, self.units:2*self.units]), np.tanh(z[:, 2*self.units:3*self.units]), self._sigmoid(z[:, 3*self.units:])
            self.c_states[:, t+1, :] = f * self.c_states[:, t, :] + i * c_tilde
            self.h_states[:, t+1, :] = o * np.tanh(self.c_states[:, t+1, :])
        return self.h_states[:, 1:, :] if self.return_sequences else self.h_states[:, -1, :]

    def backward(self, grad_output):
        batch, timesteps, _ = self.inputs.shape
        d_W, d_b, d_inputs = np.zeros_like(self.params['W']), np.zeros_like(self.params['b']), np.zeros_like(self.inputs)
        dh_next, dc_next = np.zeros((batch, self.units)), np.zeros((batch, self.units))
        
        for t in range(timesteps - 1, -1, -1):
            dh = (grad_output[:, t, :] if self.return_sequences else (grad_output if t == timesteps - 1 else 0)) + dh_next
            z = self.gates[:, t, :]
            i, f, c_tilde, o = self._sigmoid(z[:, :self.units]), self._sigmoid(z[:, self.units:2*self.units]), np.tanh(z[:, 2*self.units:3*self.units]), self._sigmoid(z[:, 3*self.units:])
            tanh_c = np.tanh(self.c_states[:, t+1, :])
            do, dc = dh * tanh_c, dh * o * (1 - tanh_c**2) + dc_next
            df, di, dc_tilde = dc * self.c_states[:, t, :], dc * c_tilde, dc * i
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dc_tilde * (1 - c_tilde**2), do * o * (1 - o)], axis=1)
            concat = np.concatenate([self.inputs[:, t, :], self.h_states[:, t, :]], axis=1)
            d_W += np.dot(concat.T, dz)
            d_b += np.sum(dz, axis=0)
            d_concat = np.dot(dz, self.params['W'].T)
            d_inputs[:, t, :], dh_next, dc_next = d_concat[:, :self.features], d_concat[:, self.features:], f * dc
        self.grads['W'], self.grads['b'] = d_W, d_b
        return d_inputs
