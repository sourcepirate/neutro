from functools import reduce

from ..base import Layer
from neutro.autograd import Tensor
from neutro.autograd import ops as autograd_ops


def _to_tensor(x):
    """Coerce `x` into an autograd Tensor if it is not already one."""
    return Tensor(x) if not isinstance(x, Tensor) else x


class _ReduceBase(Layer):
    """Shared base for merging layers that reduce a list of tensors.

    All reduce-type layers (Add, Multiply, Average, Maximum, Minimum)
    share identical build/output-shape logic and a forward pass that
    folds a single binary operation over the input list.
    """

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def _combine(self, left, right):
        """Combine two tensors. Must be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, inputs, training=False):
        if not isinstance(inputs, list):
            return inputs
        return reduce(self._combine, map(_to_tensor, inputs))


class Add(_ReduceBase):
    def _combine(self, left, right):
        return left + right


class Concatenate(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            return input_shape
        out_shape = list(input_shape[0])
        concat_dim = 0
        for shape in input_shape:
            dim = shape[self.axis]
            if dim is None:
                concat_dim = None
                break
            concat_dim += dim
        if concat_dim is not None:
            out_shape[self.axis] = concat_dim
        return tuple(out_shape)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        if not isinstance(inputs, list):
            return inputs
        tensors = [_to_tensor(i) for i in inputs]
        return autograd_ops.concatenate(tensors, axis=self.axis)


class Multiply(_ReduceBase):
    def _combine(self, left, right):
        return left * right


class Average(_ReduceBase):
    def forward(self, inputs, training=False):
        result = super().forward(inputs, training=training)
        if isinstance(inputs, list) and len(inputs) > 1:
            result = result / float(len(inputs))
        return result

    def _combine(self, left, right):
        return left + right


class Maximum(_ReduceBase):
    def _combine(self, left, right):
        return autograd_ops.maximum(left, right)


class Minimum(_ReduceBase):
    def _combine(self, left, right):
        return -autograd_ops.maximum(-left, -right)