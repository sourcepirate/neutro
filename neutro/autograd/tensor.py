"""Tensor: numpy-backed container with autograd metadata."""

import numpy as np


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def as_tensor(x):
    """Convert ``x`` to :class:`Tensor`, handling common Python / numpy types."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (list, tuple)):
        return [as_tensor(i) for i in x]
    if isinstance(x, np.ndarray):
        return Tensor(x)
    return Tensor(np.asarray(x))


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class Tensor:
    """Thin wrapper around a ``float64`` ndarray.

    Attributes:
        data: contiguous float64 array.
        grad: gradient array (or ``None``).
    """

    __slots__ = ("data", "grad")

    # -- Creation -----------------------------------------------------------

    def __init__(self, data):
        if data is None:
            raise ValueError("Tensor data cannot be None")

        arr = np.asarray(data)
        if arr.dtype != np.float64:
            arr = arr.astype(float, copy=False)
        else:
            # Ensure contiguous, non-subclass array without unnecessary copy
            arr = np.array(arr, dtype=float, copy=False, subok=False)

        self.data = arr
        self.grad = None

    # -- Magic methods ------------------------------------------------------

    def __hash__(self):
        # Identity hash: Tensor overrides __eq__ to return an array.
        return object.__hash__(self)

    def __array__(self, dtype=None):
        if dtype is None:
            return self.data
        return self.data.astype(dtype, copy=False)

    def __bool__(self):
        # Only scalar / single-element tensors have a truth value.
        if self.data.size == 1:
            return bool(self.data.item())
        raise ValueError(
            "The truth value of a Tensor with more than one element is ambiguous. "
            "Use `tensor.data.size` or `tensor.data.any()` / `tensor.data.all()`."
        )

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"

    def __format__(self, fmt):
        return format(self.data, fmt)

    def __len__(self):
        if self.data.shape == ():
            raise TypeError("len() of unsized object Tensor with scalar shape")
        return self.data.shape[0]

    def __iter__(self):
        # Iteration returns raw numpy scalars; differentiable slicing via __getitem__
        return iter(self.data)

    def __getitem__(self, idx):
        from .ops import slice_op
        return slice_op(self, idx)

    def __setitem__(self, idx, value):
        val = value.data if isinstance(value, Tensor) else value
        self.data[idx] = val
        self.grad = None  # mutation invalidates cached grad

    # Comparisons return numpy arrays (not Tensor) for ergonomics.
    def __eq__(self, other):
        other = other.data if isinstance(other, Tensor) else other
        return np.equal(self.data, np.asarray(other))

    def __ne__(self, other):
        other = other.data if isinstance(other, Tensor) else other
        return np.not_equal(self.data, np.asarray(other))

    def __lt__(self, other):
        other = other.data if isinstance(other, Tensor) else other
        return self.data < np.asarray(other)

    def __gt__(self, other):
        other = other.data if isinstance(other, Tensor) else other
        return self.data > np.asarray(other)

    def __le__(self, other):
        other = other.data if isinstance(other, Tensor) else other
        return self.data <= np.asarray(other)

    def __ge__(self, other):
        other = other.data if isinstance(other, Tensor) else other
        return self.data >= np.asarray(other)

    # -- Properties ---------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        from .ops import transpose
        return transpose(self)

    # -- Utilities ----------------------------------------------------------

    def zero_grad(self):
        self.grad = None

    def copy(self):
        return Tensor(self.data.copy())

    def astype(self, dtype):
        return Tensor(self.data.astype(dtype))

    # -- Autograd-backed ops ------------------------------------------------

    def reshape(self, *shape):
        from .ops import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def transpose(self, *axes):
        from .ops import transpose
        if axes and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return transpose(self, axes)

    def sum(self, axis=None, keepdims=False, dtype=None, out=None):
        from .ops import _sum
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False, dtype=None, out=None):
        from .ops import _mean
        return _mean(self, axis=axis, keepdims=keepdims)

    def sqrt(self):
        from .ops import sqrt
        return sqrt(self)

    def pow(self, exponent):
        from .ops import _pow
        return _pow(self, exponent)

    # Arithmetic ------------------------------------------------------------

    def __add__(self, other):
        from .ops import add
        return add(self, other)

    def __radd__(self, other):
        from .ops import add
        return add(other, self)

    def __sub__(self, other):
        from .ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        from .ops import sub
        return sub(other, self)

    def __mul__(self, other):
        from .ops import mul
        return mul(self, other)

    def __rmul__(self, other):
        from .ops import mul
        return mul(other, self)

    def __truediv__(self, other):
        from .ops import div
        return div(self, other)

    def __rtruediv__(self, other):
        from .ops import div
        return div(other, self)

    def __neg__(self):
        from .ops import neg
        return neg(self)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .ops import matmul
        return matmul(other, self)

    def __pow__(self, power):
        from .ops import _pow
        return _pow(self, power)
