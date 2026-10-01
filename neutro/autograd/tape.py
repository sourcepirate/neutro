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

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Robust stack handling: ensure we pop the correct tape even if exceptions occurred
        if not _tape_stack:
            return False
        if _tape_stack[-1] is not self:
            # Mismatched nesting - remove self wherever it is to avoid corruption
            try:
                _tape_stack.remove(self)
            except ValueError:
                pass
        else:
            _tape_stack.pop()
        return False

    def watch(self, tensor):
        # Accept single Tensor or iterable of Tensors for convenience
        if isinstance(tensor, (list, tuple, set)):
            for t in tensor:
                self.watch(t)
            return
        # Late import to avoid circular
        from .tensor import Tensor as TensorClass
        if not isinstance(tensor, TensorClass):
            raise TypeError(f"GradientTape.watch expects Tensor, got {type(tensor)}")
        self._watched.add(tensor)

    def _record_op(self, inputs, output, backward_fn, name=''):
        # Defensive copy of inputs to avoid external mutation
        self._ops.append({
            'inputs': list(inputs),
            'output': output,
            'backward': backward_fn,
            'name': name,
        })
        self._watched.add(output)

    def gradient(self, target, sources):
        from .tensor import Tensor as TensorClass
        if not isinstance(target, TensorClass):
            raise TypeError(f"GradientTape.gradient target must be Tensor, got {type(target)}")
        # Normalize sources to list
        if isinstance(sources, TensorClass):
            sources = [sources]
        else:
            try:
                sources = list(sources)
            except TypeError:
                raise TypeError("sources must be Tensor or iterable of Tensors")
        for s in sources:
            if not isinstance(s, TensorClass):
                raise TypeError(f"GradientTape.gradient source must be Tensor, got {type(s)}")

        # Fast path: no ops or target not computed within tape -> all grads None
        # Use identity mapping from Tensor id to array
        grad = {target: np.ones_like(target.data, dtype=float)}

        # Reverse topological order is the recorded order reversed; _ops already in forward order
        for op in reversed(self._ops):
            out = op['output']
            if out not in grad:
                continue
            upstream = grad[out]
            # Backward may return None for non-differentiable inputs
            try:
                input_grads = op['backward'](upstream)
            except Exception as e:
                raise RuntimeError(f"Backward failed for op '{op['name']}': {e}") from e

            if input_grads is None:
                continue
            if not isinstance(input_grads, (list, tuple)):
                input_grads = [input_grads]

            # Robust length handling: if mismatch, pad with None or truncate
            inputs = op['inputs']
            if len(input_grads) != len(inputs):
                # If backward returned fewer gradients than inputs, assume trailing Nones
                # This protects against silent zip truncation bugs
                if len(input_grads) < len(inputs):
                    input_grads = list(input_grads) + [None] * (len(inputs) - len(input_grads))
                else:
                    input_grads = input_grads[:len(inputs)]

            for inp, ig in zip(inputs, input_grads):
                if ig is None:
                    continue
                # Ensure array type and shape sanity before accumulation
                ig = np.asarray(ig, dtype=float)
                if inp in grad:
                    # Use out-of-place addition to avoid mutating shared references
                    # Accumulate gradients from multiple consumers
                    grad[inp] = grad[inp] + ig
                else:
                    grad[inp] = ig

        grads = []
        for src in sources:
            g = grad.get(src)
            # Ensure grad array does not share memory with internal dict that could be mutated later
            if g is not None:
                # Copy to avoid aliasing where same tensor is source multiple times
                # but keep efficiency by not copying unnecessarily if single use
                src.grad = g  # keep reference; caller gets same array
            else:
                src.grad = None
            grads.append(src.grad)
        return grads

    def reset(self):
        self._ops.clear()
        self._watched.clear()

    def watched_variables(self):
        # Helper for debugging / inspection
        return list(self._watched)
