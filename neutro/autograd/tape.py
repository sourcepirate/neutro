"""GradientTape: record ops and compute gradients via reverse-mode autodiff."""

import numpy as np

_tape_stack: list["GradientTape"] = []


def get_active_tape():
    """Return the currently active tape, or ``None``."""
    return _tape_stack[-1] if _tape_stack else None


class GradientTape:
    """Context manager that records differentiable ops."""

    def __init__(self):
        self._ops: list[dict] = []
        self._watched: set = set()

    # -- Context manager ----------------------------------------------------

    def __enter__(self):
        _tape_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _tape_stack:
            return False
        if _tape_stack[-1] is not self:
            # Unbalanced nesting (e.g. exception): remove ourselves gracefully.
            try:
                _tape_stack.remove(self)
            except ValueError:
                pass
        else:
            _tape_stack.pop()
        return False

    # -- Recording ----------------------------------------------------------

    def watch(self, tensor):
        """Mark ``tensor`` (or iterable of tensors) as watched."""
        if isinstance(tensor, (list, tuple, set)):
            for item in tensor:
                self.watch(item)
            return

        from .tensor import Tensor as TensorClass
        if not isinstance(tensor, TensorClass):
            raise TypeError(f"GradientTape.watch expects Tensor, got {type(tensor)}")
        self._watched.add(tensor)

    def _record_op(self, inputs, output, backward_fn, name=""):
        self._ops.append({
            "inputs": list(inputs),  # defensive copy
            "output": output,
            "backward": backward_fn,
            "name": name,
        })
        self._watched.add(output)

    # -- Differentiation ----------------------------------------------------

    def gradient(self, target, sources):
        """Compute ``d target / d sources``.

        Returns a list aligned with ``sources`` (entries ``None`` if no path).
        """
        from .tensor import Tensor as TensorClass

        sources = self._normalize_sources(sources, TensorClass)
        if not isinstance(target, TensorClass):
            raise TypeError(f"gradient target must be Tensor, got {type(target)}")

        # Seed gradient: d target / d target = 1
        grad_table: dict = {target: np.ones_like(target.data, dtype=float)}

        for op in reversed(self._ops):
            output = op["output"]
            if output not in grad_table:
                continue

            upstream = grad_table[output]

            try:
                input_grads = op["backward"](upstream)
            except Exception as exc:
                raise RuntimeError(f"Backward failed for op '{op['name']}': {exc}") from exc

            if input_grads is None:
                continue
            if not isinstance(input_grads, (list, tuple)):
                input_grads = [input_grads]

            input_grads = self._align_grad_lengths(input_grads, op["inputs"])
            self._accumulate_grads(grad_table, op["inputs"], input_grads)

        return self._collect_source_grads(grad_table, sources)

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _normalize_sources(sources, tensor_cls):
        if isinstance(sources, tensor_cls):
            return [sources]
        try:
            sources = list(sources)
        except TypeError as exc:
            raise TypeError("sources must be Tensor or iterable of Tensors") from exc
        for src in sources:
            if not isinstance(src, tensor_cls):
                raise TypeError(f"gradient source must be Tensor, got {type(src)}")
        return sources

    @staticmethod
    def _align_grad_lengths(grads, inputs):
        """Pad/truncate ``grads`` to match ``inputs``."""
        if len(grads) == len(inputs):
            return grads
        if len(grads) < len(inputs):
            return list(grads) + [None] * (len(inputs) - len(grads))
        return list(grads[:len(inputs)])

    @staticmethod
    def _accumulate_grads(grad_table, inputs, input_grads):
        for inp, ig in zip(inputs, input_grads):
            if ig is None:
                continue
            ig = np.asarray(ig, dtype=float)
            if inp in grad_table:
                grad_table[inp] = grad_table[inp] + ig  # out-of-place to avoid aliasing
            else:
                grad_table[inp] = ig

    @staticmethod
    def _collect_source_grads(grad_table, sources):
        grads = []
        for src in sources:
            grad = grad_table.get(src)
            src.grad = grad
            grads.append(grad)
        return grads

    # -- Utilities ----------------------------------------------------------

    def reset(self):
        self._ops.clear()
        self._watched.clear()

    def watched_variables(self):
        return list(self._watched)
