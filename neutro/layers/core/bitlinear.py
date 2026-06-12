import numpy as np
from ..base import Layer
from ...initializers import get as get_initializer


def weight_quantize_b1(W, eps=1e-6):
    alpha = np.mean(W)
    W_centered = W - alpha
    W_bin = np.where(W_centered > 0, 1.0, -1.0)
    beta = np.mean(np.abs(W)) + eps
    return W_bin, beta


def weight_quantize_b158(W, eps=1e-6):
    gamma = np.mean(np.abs(W)) + eps
    W_scaled = W / gamma
    W_tern = np.clip(np.round(W_scaled), -1, 1)
    beta = np.mean(np.abs(W)) + eps
    return W_tern, beta


def activation_quantize(x, bits=8, per_token=False, eps=1e-6):
    Q_b = 2 ** (bits - 1)
    if per_token:
        abs_max = np.max(np.abs(x), axis=-1, keepdims=True)
        abs_max = np.clip(abs_max, eps, None)
        gamma = abs_max
    else:
        gamma = np.max(np.abs(x))
        if gamma < eps:
            gamma = eps
    x_scaled = x * Q_b / gamma
    x_quant = np.clip(np.round(x_scaled), -Q_b + 1, Q_b - 1)
    return x_quant, gamma


class BitLinear(Layer):
    def __init__(self, units, mode='b1.58', activation_bits=8, use_bias=False, per_token=False, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.activation_bits = activation_bits
        self.use_bias = use_bias
        self.per_token = per_token
        self.kernel_initializer = get_initializer(kernel_initializer)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.params['W'] = self.kernel_initializer((self.input_dim, self.units))
        if self.use_bias:
            self.params['b'] = np.zeros((self.units,))
        self.params['gamma_ln'] = np.ones(self.input_dim)
        self.params['beta_ln'] = np.zeros(self.input_dim)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [self.units])

    def _layernorm_forward(self, x):
        self.ln_x = x
        self.ln_mean = np.mean(x, axis=-1, keepdims=True)
        self.ln_var = np.var(x, axis=-1, keepdims=True)
        self.ln_x_norm = (x - self.ln_mean) / np.sqrt(self.ln_var + 1e-6)
        return self.params['gamma_ln'] * self.ln_x_norm + self.params['beta_ln']

    def _layernorm_backward(self, grad_output):
        N = grad_output.shape[-1]
        self.grads['gamma_ln'] = np.sum(grad_output * self.ln_x_norm, axis=tuple(range(len(grad_output.shape) - 1)))
        self.grads['beta_ln'] = np.sum(grad_output, axis=tuple(range(len(grad_output.shape) - 1)))
        dx_norm = grad_output * self.params['gamma_ln']
        std_inv = 1.0 / np.sqrt(self.ln_var + 1e-6)
        dx = (1.0 / N) * std_inv * (N * dx_norm - np.sum(dx_norm, axis=-1, keepdims=True) - self.ln_x_norm * np.sum(dx_norm * self.ln_x_norm, axis=-1, keepdims=True))
        return dx

    def _weight_quantize(self, W):
        if self.mode == 'b1':
            return weight_quantize_b1(W)
        elif self.mode == 'b1.58':
            return weight_quantize_b158(W)

    def _activation_quantize(self, x):
        return activation_quantize(x, bits=self.activation_bits, per_token=self.per_token)

    def forward(self, inputs, training=False):
        self.inputs = inputs

        x_norm = self._layernorm_forward(inputs)

        x_q, self.gamma = self._activation_quantize(x_norm)

        W_q, self.beta = self._weight_quantize(self.params['W'])

        self.W_q = W_q
        self.x_q = x_q
        self.Q_b = 2 ** (self.activation_bits - 1)
        self.deq_scale = self.beta * self.gamma / self.Q_b

        y = (x_q @ W_q) * self.deq_scale
        if self.use_bias:
            y += self.params['b']
        return y

    def backward(self, grad_output):
        if self.use_bias:
            self.grads['b'] = np.sum(grad_output, axis=tuple(range(len(grad_output.shape) - 1)))

        inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])

        self.grads['W'] = inputs_flat.T @ grad_flat

        dx = grad_flat @ self.params['W'].T
        dx = dx.reshape(self.inputs.shape)

        dx = self._layernorm_backward(dx)
        return dx
