"""Function: base class for differentiable ops."""

import numpy as np

from .tape import get_active_tape
from .tensor import Tensor
from .utils import broadcast_backward


class _Ctx:
    """Context to stash tensors / metadata for the backward pass."""

    def __init__(self):
        self.saved_data: dict = {}

    def save_for_backward(self, **kwargs):
        self.saved_data.update(kwargs)

    def get_saved(self, key, default=None):
        return self.saved_data.get(key, default)


class Function:
    """Base class. Subclasses implement ``forward(ctx, *data)`` and ``backward(ctx, g)``."""

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = _Ctx()
        tensor_args, data_args = _split_tensor_args(args)
        tensor_kwargs, data_kwargs = _split_tensor_kwargs(kwargs)
        all_tensor_inputs = tensor_args + tensor_kwargs

        output_data = _normalize_output(cls.forward(ctx, *data_args, **data_kwargs))
        result = Tensor(output_data)

        tape = get_active_tape()
        if tape is not None and all_tensor_inputs and _should_record(tape, all_tensor_inputs):
            backward_ctx = _Ctx()
            backward_ctx.saved_data = _build_saved_copy(ctx.saved_data)
            backward_fn = _make_backward_fn(
                cls, backward_ctx, tensor_args, tensor_kwargs, all_tensor_inputs
            )
            tape._record_op(all_tensor_inputs, result, backward_fn, cls.__name__)

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_tensor_args(args):
    tensors = [arg for arg in args if isinstance(arg, Tensor)]
    data = [arg.data if isinstance(arg, Tensor) else arg for arg in args]
    return tensors, data


def _split_tensor_kwargs(kwargs):
    tensors = [value for value in kwargs.values() if isinstance(value, Tensor)]
    data = {
        key: value.data if isinstance(value, Tensor) else value
        for key, value in kwargs.items()
    }
    return tensors, data


def _normalize_output(data):
    if isinstance(data, Tensor):
        return data.data
    return np.asarray(data)


def _should_record(tape, inputs):
    return any(t in tape._watched for t in inputs)


def _build_saved_copy(saved_data):
    return {
        name: value.data if isinstance(value, Tensor) else value
        for name, value in saved_data.items()
    }


def _make_backward_fn(cls, ctx, tensor_args, tensor_kwargs, all_inputs):
    tensor_args = list(tensor_args)
    tensor_kwargs = list(tensor_kwargs)
    all_inputs = list(all_inputs)

    def backward(upstream):
        grads = cls.backward(ctx, upstream)

        if grads is None:
            return [None] * len(all_inputs)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]

        grads = _align_backward_grads(grads, tensor_args, tensor_kwargs, all_inputs)
        return [
            _broadcast_grad(tensor, grad)
            for tensor, grad in zip(all_inputs, grads)
        ]

    return backward


def _broadcast_grad(tensor, grad):
    if grad is None:
        return None
    array = np.asarray(grad)
    return array if array.shape == tensor.shape else broadcast_backward(array, tensor.shape)


def _align_backward_grads(grads, tensor_args, tensor_kwargs, all_inputs):
    grads = list(grads)
    expected = len(all_inputs)
    if len(grads) == expected:
        return grads
    if len(grads) == len(tensor_args) and tensor_kwargs:
        return grads + [None] * len(tensor_kwargs)
    return (grads + [None] * expected)[:expected]
