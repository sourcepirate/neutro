import numpy as np
from .tensor import Tensor
from .tape import get_active_tape
from .utils import broadcast_backward


class _Ctx:
    def __init__(self):
        self.saved_data = {}

    def save_for_backward(self, **kwargs):
        # Shallow copy of kwargs; values are expected to be numpy arrays/scalars
        # Update instead of replace to allow multiple calls
        self.saved_data.update(kwargs)

    def get_saved(self, key, default=None):
        return self.saved_data.get(key, default)


class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = _Ctx()

        # Separate tensor vs non-tensor args
        tensor_args = []
        tensor_arg_indices = []  # positions of tensor args in args
        data_args = []
        for idx, a in enumerate(args):
            if isinstance(a, Tensor):
                tensor_args.append(a)
                tensor_arg_indices.append(idx)
                data_args.append(a.data)
            else:
                data_args.append(a)

        # Handle Tensor kwargs: collect them for gradient tracking
        tensor_kwarg_keys = []
        tensor_kwargs_list = []
        data_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                tensor_kwarg_keys.append(k)
                tensor_kwargs_list.append(v)
                data_kwargs[k] = v.data
            else:
                data_kwargs[k] = v

        # Combine all tensor inputs for tape bookkeeping
        all_tensor_inputs = tensor_args + tensor_kwargs_list

        # Forward
        output_data = cls.forward(ctx, *data_args, **data_kwargs)
        # Normalize output_data to numpy array (handles scalars)
        if isinstance(output_data, Tensor):
            # If user accidentally returns Tensor, extract data
            output_data = output_data.data
        else:
            output_data = np.asarray(output_data)

        result = Tensor(output_data)

        tape = get_active_tape()
        # Only record if tape is active and any input is watched
        if tape is not None and all_tensor_inputs:
            watched = tape._watched
            should_record = any(t in watched for t in all_tensor_inputs)
            if should_record:
                # Prepare saved_data copy: convert any remaining Tensor values to numpy,
                # and ensure we don't mutate original ctx that might be used elsewhere.
                # Also copy arrays that are views to avoid aliasing issues if forward mutates later.
                saved_copy = {}
                for name, val in ctx.saved_data.items():
                    if isinstance(val, Tensor):
                        saved_copy[name] = val.data
                    elif isinstance(val, np.ndarray):
                        # Keep reference but ensure not a view that could be overwritten?
                        # We keep reference without copy for efficiency; if op needs copy it should do so in forward
                        saved_copy[name] = val
                    else:
                        saved_copy[name] = val

                ctx_copy = _Ctx()
                ctx_copy.saved_data = saved_copy

                # Capture for closure to avoid late binding issues
                _tensor_args = list(tensor_args)
                _tensor_kwargs_list = list(tensor_kwargs_list)
                _tensor_arg_indices = list(tensor_arg_indices)
                _tensor_kwarg_keys = list(tensor_kwarg_keys)
                _all_inputs = list(all_tensor_inputs)
                _cls = cls

                def bw(g):
                    # g is upstream gradient (numpy array)
                    grad_inputs = _cls.backward(ctx_copy, g)
                    if grad_inputs is None:
                        return [None] * len(_all_inputs)
                    if not isinstance(grad_inputs, (list, tuple)):
                        grad_inputs = [grad_inputs]
                    # grad_inputs corresponds to *tensor_args in order plus kwargs?
                    # For robustness: if backward returns fewer than expected, pad with None
                    # Expectation: backward returns gradients for each Tensor input in order of appearance:
                    # first positional tensor_args, then kwargs tensors (in order of tensor_kwarg_keys)
                    expected_len = len(_all_inputs)
                    if len(grad_inputs) != expected_len:
                        # Common case: op advertises gradients only for positional args
                        # If mismatch, attempt to align: if only positional count matches, pad kwargs with None
                        if len(grad_inputs) == len(_tensor_args) and _tensor_kwargs_list:
                            # Extend with Nones for kwargs (assume non-differentiable kwargs like indices)
                            grad_inputs = list(grad_inputs) + [None] * len(_tensor_kwargs_list)
                        elif len(grad_inputs) < expected_len:
                            grad_inputs = list(grad_inputs) + [None] * (expected_len - len(grad_inputs))
                        else:
                            grad_inputs = grad_inputs[:expected_len]

                    out_grads = []
                    # Broadcast each grad back to its original shape
                    for t, gi in zip(_all_inputs, grad_inputs):
                        if gi is None:
                            out_grads.append(None)
                        else:
                            gi_arr = np.asarray(gi)
                            # If shapes already match, avoid extra copy
                            if gi_arr.shape != t.shape:
                                gi_arr = broadcast_backward(gi_arr, t.shape)
                            out_grads.append(gi_arr)
                    return out_grads

                tape._record_op(_all_inputs, result, bw, cls.__name__)

        return result
