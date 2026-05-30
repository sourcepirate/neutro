import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer

class SimpleRNN(Layer):
    def __init__(self, units, activation='tanh', return_sequences=False):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.activation_name = activation

    def build(self, input_shape):
        self.features = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['Wx'] = init((self.features, self.units))
        self.params['Wh'] = init((self.units, self.units))
        self.params['b'] = get_initializer('zeros')((self.units,))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)

    def forward(self, inputs, training=False):
        self.inputs = inputs
        batch, timesteps, _ = inputs.shape
        self.h_states = np.zeros((batch, timesteps + 1, self.units))
        
        for t in range(timesteps):
            z = np.dot(inputs[:, t, :], self.params['Wx']) + np.dot(self.h_states[:, t, :], self.params['Wh']) + self.params['b']
            if self.activation_name == 'tanh':
                self.h_states[:, t+1, :] = np.tanh(z)
            else:
                self.h_states[:, t+1, :] = z 
        if self.return_sequences:
            return self.h_states[:, 1:, :]
        return self.h_states[:, -1, :]

    def backward(self, grad_output):
        batch, timesteps, _ = self.inputs.shape
        d_Wx, d_Wh, d_b = np.zeros_like(self.params['Wx']), np.zeros_like(self.params['Wh']), np.zeros_like(self.params['b'])
        d_inputs = np.zeros_like(self.inputs)
        dh_next = np.zeros((batch, self.units))
        
        for t in range(timesteps - 1, -1, -1):
            dh = (grad_output[:, t, :] if self.return_sequences else (grad_output if t == timesteps - 1 else 0)) + dh_next
            dz = dh * (1 - self.h_states[:, t+1, :]**2)
            d_Wx += np.dot(self.inputs[:, t, :].T, dz)
            d_Wh += np.dot(self.h_states[:, t, :].T, dz)
            d_b += np.sum(dz, axis=0)
            d_inputs[:, t, :] = np.dot(dz, self.params['Wx'].T)
            dh_next = np.dot(dz, self.params['Wh'].T)
            
        self.grads['Wx'], self.grads['Wh'], self.grads['b'] = d_Wx, d_Wh, d_b
        return d_inputs
