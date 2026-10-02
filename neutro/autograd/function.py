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

        # Split Tensor vs non-Tensor inputs
        tensor_args, tensor_arg_indices, data_args = _split_tensor_args(args)
        tensor_kwarg_keys, tensor_kwargs, data_kwargs = _split_tensor_kwargs(kwargs)

        all_tensor_inputs = tensor_args + tensor_kwargs

        # Forward pass on raw numpy data
        output_data = cls.forward(ctx, *data_args, **data_kwargs)
        output_data = _normalize_output(output_data)
        result = Tensor(output_data)

        # Record for reverse-mode if needed
        tape = get_active_tape()
        if tape is not None and all_tensor_inputs and _should_record(tape, all_tensor_inputs):
            saved = _build_saved_copy(ctx.saved_data)
            ctx_copy = _Ctx()
            ctx_copy.saved_data = saved

            backward_fn = _make_backward_fn(
                cls, ctx_copy,
                tensor_args, tensor_kwargs,
                tensor_arg_indices, tensor_kwarg_keys,
                all_tensor_inputs,
            )
            tape._record_op(all_tensor_inputs, result, backward_fn, cls.__name__)

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_tensor_args(args):
    tensor_args = []
    indices = []
    data_args = []
    for idx, arg in enumerate(args):
        if isinstance(arg, Tensor):
            tensor_args.append(arg)
            indices.append(idx)
            data_args.append(arg.data)
        else:
            data_args.append(arg)
    return tensor_args, indices, data_args


def _split_tensor_kwargs(kwargs):
    keys, tensors, data_kwargs = [], [], {}
    for key, val in kwargs.items():
        if isinstance(val, Tensor):
            keys.append(key)
            tensors.append(val)
            data_kwargs[key] = val.data
        else:
            data_kwargs[key] = val
    return keys, tensors, data_kwargs


def _normalize_output(data):
    if isinstance(data, Tensor):
        return data.data
    return np.asarray(data)


def _should_record(tape, inputs):
    return any(t in tape._watched for t in inputs)


def _build_saved_copy(saved_data):
    copy = {}
    for name, val in saved_data.items():
        if isinstance(val, Tensor):
            copy[name] = val.data
        else:
            copy[name] = val
    return copy


def _make_backward_fn(cls, ctx_copy, tensor_args, tensor_kwargs,
                      tensor_arg_indices, tensor_kwarg_keys, all_inputs):
    # Capture lists to avoid late-binding issues
    tensor_args = list(tensor_args)
    tensor_kwargs = list(tensor_kwargs)
    all_inputs = list(all_inputs)

    def backward(upstream):
        grads = cls.backward(ctx_copy, upstream)

        if grads is None:
            return [None] * len(all_inputs)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]

        grads = _align_backward_grads(grads, tensor_args, tensor_kwargs, all_inputs)

        # Broadcast each gradient to its input shape
        out = []
        for tensor, grad in zip(all_inputs, grads):
            if grad is None:
                out.append(None)
            else:
                arr = np.asarray(grad)
                if arr.shape != tensor.shape:
                    arr = broadcast_backward(arr, tensor.shape)
                out.append(arr)
        return out

    return backward


def _align_backward_grads(grads, tensor_args, tensor_kwargs, all_inputs):
    expected = len(all_inputs)
    if len(grads) == expected:
        return grads
    # Common case: op returns grads only for positional args
    if len(grads) == len(tensor_args) and tensor_kwargs:
        return list(grads) + [None] * len(tensor_kwargs)
    if len(grads) < expected:
        return list(grads) + [None] * (expected - len(grads))
    return list(grads[:expected])
