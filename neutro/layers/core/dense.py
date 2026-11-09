from ..base import Layer
from ...initializers import get as get_initializer
from ...activations import get as get_activation
from neutro.autograd import Tensor


class Dense(Layer):
    """A fully-connected layer.

    Computes ``y = activation(x @ W + b)`` on the last axis.

    Shapes::

        x: (B, D) or (B, S, D)
        W: (D, units)
        b: (units,)
        y: (B, units) or (B, S, units)

    The layer only implements ``forward``. The base ``Layer.backward`` uses
    ``GradientTape`` to compute ``dL/dW``, ``dL/db``, and ``dL/dx``.
    """

    def __init__(self, units: int, activation=None, use_bias: bool = True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

    def build(self, input_shape) -> None:
        """Allocate ``W`` of shape ``(input_dim, units)`` and ``b`` of shape ``(units,)``."""
        self.input_dim = input_shape[-1]
        self.params['W'] = Tensor(self.kernel_initializer((self.input_dim, self.units)))
        if self.use_bias:
            self.params['b'] = Tensor(self.bias_initializer((self.units,)))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Replace the last dimension of ``input_shape`` with ``units``."""
        return (*input_shape[:-1], self.units)

    def forward(self, inputs, training=False):
        """Apply ``inputs @ W + b`` and the optional activation.

        Args:
            inputs: Tensor of shape ``(..., input_dim)``.
            training: Accepted for API compatibility; Dense has no
                training-dependent behavior.

        Returns:
            Tensor of shape ``(..., units)``.
        """
        z = inputs @ self.params['W']
        if self.use_bias:
            z = z + self.params['b']
        if self.activation:
            return self.activation(z)
        return z
