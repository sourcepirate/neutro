import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer

class GRU(Layer):
    """
    Gated Recurrent Unit (GRU) layer.
    
    Args:
        units: Positive integer, dimensionality of the output space.
        return_sequences: Boolean. Whether to return the last output in the 
            output sequence, or the full sequence.
    """
    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.features = input_shape[-1]
        init = get_initializer('glorot_uniform')
        
        # We'll use separate weights for the gates and the candidate to make reset gate logic easier
        # Wz, Wr, Wh for inputs
        self.params['W'] = init((self.features, 3 * self.units))
        # Uz, Ur, Uh for recurrent
        self.params['U'] = init((self.units, 3 * self.units))
        # biases
        self.params['b'] = get_initializer('zeros')((3 * self.units,))
        
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        return (input_shape[0], self.units)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, inputs, training=False):
        self.inputs = inputs
        batch, timesteps, _ = inputs.shape
        self.h_states = np.zeros((batch, timesteps + 1, self.units))
        
        # Store intermediate values for backward pass
        self.z_gates = np.zeros((batch, timesteps, self.units))
        self.r_gates = np.zeros((batch, timesteps, self.units))
        self.h_tilde = np.zeros((batch, timesteps, self.units))
        self.x_W = np.dot(inputs, self.params['W']) # (batch, timesteps, 3*units)
        
        for t in range(timesteps):
            x_W_t = self.x_W[:, t, :] + self.params['b']
            h_prev = self.h_states[:, t, :]
            
            # Gates (z, r)
            z_r_hidden = np.dot(h_prev, self.params['U'][:, :2*self.units])
            z_r_logits = x_W_t[:, :2*self.units] + z_r_hidden
            z_r = self._sigmoid(z_r_logits)
            
            z = z_r[:, :self.units]
            r = z_r[:, self.units:]
            
            self.z_gates[:, t, :] = z
            self.r_gates[:, t, :] = r
            
            # Candidate h_tilde
            h_tilde_hidden = np.dot(r * h_prev, self.params['U'][:, 2*self.units:])
            h_tilde_logits = x_W_t[:, 2*self.units:] + h_tilde_hidden
            h_tilde = np.tanh(h_tilde_logits)
            self.h_tilde[:, t, :] = h_tilde
            
            # New state
            self.h_states[:, t+1, :] = (1 - z) * h_tilde + z * h_prev
            
        if self.return_sequences:
            return self.h_states[:, 1:, :]
        return self.h_states[:, -1, :]

    def backward(self, grad_output):
        batch, timesteps, _ = self.inputs.shape
        d_W = np.zeros_like(self.params['W'])
        d_U = np.zeros_like(self.params['U'])
        d_b = np.zeros_like(self.params['b'])
        d_inputs = np.zeros_like(self.inputs)
        dh_next = np.zeros((batch, self.units))
        
        for t in range(timesteps - 1, -1, -1):
            dh = (grad_output[:, t, :] if self.return_sequences else (grad_output if t == timesteps - 1 else 0)) + dh_next
            
            z = self.z_gates[:, t, :]
            r = self.r_gates[:, t, :]
            h_tilde = self.h_tilde[:, t, :]
            h_prev = self.h_states[:, t, :]
            
            # dL/dh_t -> dL/dz, dL/dh_tilde, dL/dh_prev
            dz = dh * (h_prev - h_tilde)
            dh_tilde = dh * (1 - z)
            dh_prev_from_h = dh * z
            
            # Backprop through tanh for h_tilde
            dtanh = dh_tilde * (1 - h_tilde**2)
            
            # dL/dh_tilde -> dL/dW_h, dL/dU_h, dL/dr
            # h_tilde = tanh(x*Wh + (r*h_prev)*Uh + bh)
            d_W[:, 2*self.units:] += np.dot(self.inputs[:, t, :].T, dtanh)
            d_U[:, 2*self.units:] += np.dot((r * h_prev).T, dtanh)
            d_b[2*self.units:] += np.sum(dtanh, axis=0)
            
            dr_h_prev = np.dot(dtanh, self.params['U'][:, 2*self.units:].T)
            dr = dr_h_prev * h_prev
            dh_prev_from_tilde = dr_h_prev * r
            
            # Backprop through sigmoids for z, r
            dz_logits = dz * z * (1 - z)
            dr_logits = dr * r * (1 - r)
            dzr_logits = np.concatenate([dz_logits, dr_logits], axis=1)
            
            # dL/dzr -> dL/dW_zr, dL/dU_zr
            d_W[:, :2*self.units] += np.dot(self.inputs[:, t, :].T, dzr_logits)
            d_U[:, :2*self.units] += np.dot(h_prev.T, dzr_logits)
            d_b[:2*self.units] += np.sum(dzr_logits, axis=0)
            
            dh_prev_from_gates = np.dot(dzr_logits, self.params['U'][:, :2*self.units].T)
            
            # Total dh_prev for next step
            dh_next = dh_prev_from_h + dh_prev_from_tilde + dh_prev_from_gates
            
            # Gradient wrt inputs
            d_inputs[:, t, :] = np.dot(dzr_logits, self.params['W'][:, :2*self.units].T) + \
                                np.dot(dtanh, self.params['W'][:, 2*self.units:].T)
            
        self.grads['W'] = d_W
        self.grads['U'] = d_U
        self.grads['b'] = d_b
        return d_inputs
