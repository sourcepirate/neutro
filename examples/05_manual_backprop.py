"""Manual backprop vs the tape-backed path.

Run from the repository root::

    python examples/05_manual_backprop.py

This implements a Dense layer's chain rule by hand, then compares it with the
same layer using Neutro's default tape-backed backward pass.
"""

import numpy as np

from neutro.autograd import Tensor
from neutro.layers import Dense
from neutro.layers.base import Layer


class ManualDense(Layer):
    """Dense without autograd: forward and backward written by hand."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        rng = np.random.default_rng(0)
        self.params["W"] = Tensor(rng.normal(0, 0.1, (input_shape[-1], self.units)))
        self.params["b"] = Tensor(np.zeros(self.units))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def forward(self, inputs, training=False):
        x = inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs)
        self._x = x
        z = x @ self.params["W"].data + self.params["b"].data
        return Tensor(z)

    def backward(self, grad_output):
        g = grad_output.data if isinstance(grad_output, Tensor) else np.asarray(grad_output)
        self.grads["W"] = self._x.T @ g
        self.grads["b"] = g.sum(axis=0)
        return g @ self.params["W"].data.T


def main():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 5))
    upstream = rng.normal(size=(3, 4))

    manual = ManualDense(4)
    auto = Dense(4)
    manual.build((3, 5))
    auto.build((3, 5))

    # Give both layers identical weights.
    auto.params["W"].data = manual.params["W"].data.copy()
    auto.params["b"].data = manual.params["b"].data.copy()

    y_manual = manual(x)
    y_auto = auto(x)

    dx_manual = manual.backward(upstream)
    dx_auto = auto.backward(upstream)

    print("Forward difference:      ", np.max(np.abs(y_manual.data - y_auto.data)))
    print("dW difference:           ", np.max(np.abs(manual.grads["W"] - auto.grads["W"])))
    print("db difference:           ", np.max(np.abs(manual.grads["b"] - auto.grads["b"])))
    print("dInput difference:       ", np.max(np.abs(dx_manual - dx_auto)))
    print()
    print("Both paths use the same three chain-rule formulas:")
    print("  dW = x.T @ dY")
    print("  db = sum(dY, axis=0)")
    print("  dX = dY @ W.T")
    print()
    print("Next: docs/tutorials/05-manual-backprop.md")


if __name__ == "__main__":
    main()
