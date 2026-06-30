import numpy as np
from ..base import Layer
from neutro.autograd import Tensor
from neutro.autograd import ops as autograd_ops


def _to_tensor(x):
    return Tensor(x) if not isinstance(x, Tensor) else x


class Add(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.input_shape = input_shape
            self.output_shape = input_shape[0]
        else:
            self.input_shape = input_shape
            self.output_shape = input_shape
        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def forward(self, inputs, training=False):
        if not isinstance(inputs, list):
            return inputs
        result = _to_tensor(inputs[0])
        for i in range(1, len(inputs)):
            result = result + _to_tensor(inputs[i])
        return result


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


class Multiply(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        if not isinstance(inputs, list):
            return inputs
        result = _to_tensor(inputs[0])
        for i in range(1, len(inputs)):
            result = result * _to_tensor(inputs[i])
        return result


class Average(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        if not isinstance(inputs, list):
            return inputs
        result = _to_tensor(inputs[0])
        for i in range(1, len(inputs)):
            result = result + _to_tensor(inputs[i])
        return result / float(len(inputs))


class Maximum(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        if not isinstance(inputs, list):
            return inputs
        result = _to_tensor(inputs[0])
        for i in range(1, len(inputs)):
            other = _to_tensor(inputs[i])
            result = autograd_ops.maximum(result, other)
        return result


class Minimum(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def forward(self, inputs, training=False):
        if not isinstance(inputs, list):
            return inputs
        result = _to_tensor(inputs[0])
        for i in range(1, len(inputs)):
            other = _to_tensor(inputs[i])
            result = -autograd_ops.maximum(-result, -other)
        return result