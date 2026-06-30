import numpy as np
from .tensor import Tensor
from .tape import get_active_tape
from .utils import broadcast_backward


class _Ctx:
    def __init__(self):
        self.saved_data = {}

    def save_for_backward(self, **kwargs):
        self.saved_data.update(kwargs)


class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = _Ctx()

        tensor_args = []
        data_args = []
        for a in args:
            if isinstance(a, Tensor):
                tensor_args.append(a)
                data_args.append(a.data)
            else:
                data_args.append(a)

        data_kwargs = {}
        for k, v in kwargs.items():
            data_kwargs[k] = v.data if isinstance(v, Tensor) else v

        output_data = cls.forward(ctx, *data_args, **data_kwargs)
        result = Tensor(output_data)

        tape = get_active_tape()
        if tape and any(id(t) in tape._watched for t in tensor_args):
            for name in ctx.saved_data:
                val = ctx.saved_data[name]
                if isinstance(val, Tensor):
                    ctx.saved_data[name] = val.data

            ctx_copy = _Ctx()
            ctx_copy.saved_data = ctx.saved_data.copy()

            def bw(g):
                grad_inputs = cls.backward(ctx_copy, g)
                if not isinstance(grad_inputs, (list, tuple)):
                    grad_inputs = [grad_inputs]
                result = []
                for t, gi in zip(tensor_args, grad_inputs):
                    if gi is not None:
                        result.append(broadcast_backward(gi, t.shape))
                    else:
                        result.append(None)
                return result

            tape._record_op(tensor_args, result, bw, cls.__name__)

        return result
