import numpy as np


def as_tensor(x):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, list):
        return [as_tensor(i) for i in x]
    if isinstance(x, np.ndarray):
        return Tensor(x)
    return Tensor(np.asarray(x))


class Tensor:
    __slots__ = ('data', 'grad')

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
        self.grad = None

    def __array__(self, dtype=None):
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

    def zero_grad(self):
        self.grad = None

    def __eq__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return np.equal(self.data, np.asarray(other))

    def __ne__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return np.not_equal(self.data, np.asarray(other))

    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return self.data < np.asarray(other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return self.data > np.asarray(other)

    def __le__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return self.data <= np.asarray(other)

    def __ge__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return self.data >= np.asarray(other)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        from .ops import transpose
        return transpose(self)

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

    def sum(self, axis=None, keepdims=False, **kwargs):
        from .ops import _sum
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False, **kwargs):
        from .ops import _mean
        return _mean(self, axis=axis, keepdims=keepdims)

    def sqrt(self):
        from .ops import sqrt
        return sqrt(self)

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

    def pow(self, exponent):
        from .ops import _pow
        return _pow(self, exponent)

    def copy(self):
        return Tensor(self.data.copy())

    def __format__(self, fmt):
        return format(self.data, fmt)

    @property
    def size(self):
        return self.data.size

    def __pow__(self, power):
        from .ops import _pow
        return _pow(self, power)

    def __getitem__(self, idx):
        from .ops import slice_op
        return slice_op(self, idx)

    def __setitem__(self, idx, value):
        self.data[idx] = np.asarray(value) if isinstance(value, Tensor) else value
