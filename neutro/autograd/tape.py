import numpy as np

_tape_stack = []


def get_active_tape():
    return _tape_stack[-1] if _tape_stack else None


class GradientTape:
    def __init__(self):
        self._ops = []
        self._watched = set()

    def __enter__(self):
        _tape_stack.append(self)
        return self

    def __exit__(self, *args):
        _tape_stack.pop()

    def watch(self, tensor):
        self._watched.add(id(tensor))

    def _record_op(self, inputs, output, backward_fn, name=''):
        self._ops.append({
            'inputs': list(inputs),
            'output': output,
            'backward': backward_fn,
            'name': name,
        })
        self._watched.add(id(output))

    def gradient(self, target, sources):
        grad = {id(target): np.ones_like(target.data)}

        for op in reversed(self._ops):
            out_id = id(op['output'])
            if out_id not in grad:
                continue
            upstream = grad[out_id]
            input_grads = op['backward'](upstream)
            for inp, ig in zip(op['inputs'], input_grads):
                if ig is not None:
                    inp_id = id(inp)
                    if inp_id in grad:
                        grad[inp_id] += ig
                    else:
                        grad[inp_id] = ig

        for src in sources:
            src.grad = grad.get(id(src))

        return {id(s): grad.get(id(s)) for s in sources}

    def reset(self):
        self._ops.clear()
        self._watched.clear()
