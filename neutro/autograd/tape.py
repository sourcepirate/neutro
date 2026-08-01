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
        self._watched.add(tensor)

    def _record_op(self, inputs, output, backward_fn, name=''):
        self._ops.append({
            'inputs': list(inputs),
            'output': output,
            'backward': backward_fn,
            'name': name,
        })
        self._watched.add(output)

    def gradient(self, target, sources):
        grad = {target: np.ones_like(target.data)}

        for op in reversed(self._ops):
            out = op['output']
            if out not in grad:
                continue
            upstream = grad[out]
            input_grads = op['backward'](upstream)
            for inp, ig in zip(op['inputs'], input_grads):
                if ig is not None:
                    if inp in grad:
                        grad[inp] += ig
                    else:
                        grad[inp] = ig

        grads = []
        for src in sources:
            src.grad = grad.get(src)
            grads.append(src.grad)
        return grads

    def reset(self):
        self._ops.clear()
        self._watched.clear()
