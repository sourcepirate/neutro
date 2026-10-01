"""Differentiable operations built on :class:`Function`.

Each op defines a ``forward`` (numpy) and ``backward`` (gradient) pair.
Public helpers (``add``, ``matmul``, ``_sum``, …) handle ``Tensor`` wrapping
and validation.
"""

import numpy as np

from .function import Function
from .tensor import Tensor

# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _ensure_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_axis(axis, ndim):
    """Normalize ``axis`` to a tuple of non-negative, unique axes."""
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)

    axis = tuple(a if a >= 0 else a + ndim for a in axis)

    for a in axis:
        if not 0 <= a < ndim:
            raise ValueError(f"axis {a} out of bounds for ndim {ndim}")
    if len(set(axis)) != len(axis):
        raise ValueError(f"repeated axis in {axis}")
    return tuple(sorted(axis))


def _expand_grad_to_shape(grad, x_shape, axis, keepdims):
    """Broadcast ``grad`` (reduced) back to ``x_shape``.

    Used by ``sum`` / ``mean`` / ``max`` backward passes.
    """
    x_shape = tuple(x_shape)
    grad = np.asarray(grad)

    if axis is None:
        return np.broadcast_to(grad, x_shape).copy()

    axes = _normalize_axis(axis, len(x_shape))
    if keepdims:
        return np.broadcast_to(grad, x_shape).copy()

    # Need to re-insert singleton dims at reduced axes
    expanded = []
    g_idx = 0
    axes_set = set(axes)
    for dim in range(len(x_shape)):
        if dim in axes_set:
            expanded.append(1)
        else:
            expanded.append(grad.shape[g_idx])
            g_idx += 1

    return np.broadcast_to(grad.reshape(expanded), x_shape).copy()


def _swap_last_two(x):
    return np.swapaxes(x, -1, -2) if x.ndim >= 2 else x


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class _Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, g):
        return g, g


class _Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a - b

    @staticmethod
    def backward(ctx, g):
        return g, -g


class _Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a * b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data["a"], ctx.saved_data["b"]
        return g * b, g * a


class _Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a / b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data["a"], ctx.saved_data["b"]
        return g / b, -g * a / (b ** 2)


class _Neg(Function):
    @staticmethod
    def forward(ctx, x):
        return -x

    @staticmethod
    def backward(ctx, g):
        return -g


class _Pow(Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x=x, k=k)
        return x ** k

    @staticmethod
    def backward(ctx, g):
        x, k = ctx.saved_data["x"], ctx.saved_data["k"]
        if k == 0:
            return np.zeros_like(x, dtype=float) * g
        return k * (x ** (k - 1)) * g


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

def _matmul_vec_mat_left(a, b, g):
    """Gradients for ``a (K,) @ b (...,K,N) -> (...,N)``."""
    # gb[...,k,n] = a[k] * g[...,n]
    batch_shape = b.shape[:-2]
    if not batch_shape:
        ga = b @ g
        gb = np.outer(a, g)
        return ga, gb

    # Batched: accumulate ga over batch, gb per-batch outer
    gb = np.empty_like(b, dtype=float)
    ga = np.zeros_like(a, dtype=float)
    for idx in np.ndindex(batch_shape):
        b_slice = b[idx]
        g_slice = g[idx]
        gb[idx] = np.outer(a, g_slice)
        ga += b_slice @ g_slice
    return ga, gb


def _matmul_mat_vec_right(a, b, g):
    """Gradients for ``a (...,M,K) @ b (K,) -> (...,M)``."""
    batch_shape = a.shape[:-2]
    if not batch_shape:
        ga = np.outer(g, b)
        gb = a.T @ g
        return ga, gb

    gb = np.zeros_like(b, dtype=float)
    ga = np.empty_like(a, dtype=float)
    for idx in np.ndindex(batch_shape):
        a_slice = a[idx]
        g_slice = g[idx]
        ga[idx] = np.outer(g_slice, b)
        gb += a_slice.T @ g_slice
    return ga, gb


class _Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a @ b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data["a"], ctx.saved_data["b"]

        # Vector-vector
        if a.ndim == 1 and b.ndim == 1:
            return g * b, g * a

        # Matrix / batched matrix (both >=2D) – generic formula
        if a.ndim >= 2 and b.ndim >= 2:
            ga = np.matmul(g, _swap_last_two(b))
            gb = np.matmul(_swap_last_two(a), g)
            return ga, gb

        # Vector-matrix
        if a.ndim == 1:
            return _matmul_vec_mat_left(a, b, g)

        # Matrix-vector
        return _matmul_mat_vec_right(a, b, g)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

class _Sum(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        ctx.save_for_backward(x_shape=x.shape, axis=axis, keepdims=keepdims)
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x_shape = ctx.saved_data["x_shape"]
        axis = ctx.saved_data["axis"]
        keepdims = ctx.saved_data["keepdims"]
        return _expand_grad_to_shape(g, x_shape, axis, keepdims)


class _Mean(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        if axis is None:
            n = x.size
        else:
            axes = _normalize_axis(axis, x.ndim)
            n = int(np.prod([x.shape[d] for d in axes]))
            axis = axes if isinstance(axis, (tuple, list)) else axis
        ctx.save_for_backward(x_shape=x.shape, axis=axis, keepdims=keepdims, n=n)
        return np.mean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x_shape = ctx.saved_data["x_shape"]
        axis = ctx.saved_data["axis"]
        keepdims = ctx.saved_data["keepdims"]
        n = ctx.saved_data["n"]
        return _expand_grad_to_shape(g, x_shape, axis, keepdims) / n


class _Max(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        ctx.save_for_backward(x=x, axis=axis, keepdims=keepdims)
        return np.max(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x, axis, keepdims = ctx.saved_data["x"], ctx.saved_data["axis"], ctx.saved_data["keepdims"]
        x_max = np.max(x, axis=axis, keepdims=True)
        mask = (x == x_max).astype(float)

        # Split ties equally
        if axis is None:
            mask /= np.clip(mask.sum(), 1e-15, None)
            return g * mask

        s = np.clip(mask.sum(axis=axis, keepdims=True), 1e-15, None)
        mask /= s
        grad_expanded = _expand_grad_to_shape(g, x.shape, axis, keepdims)
        return grad_expanded * mask


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------

class _Transpose(Function):
    @staticmethod
    def forward(ctx, x, axes):
        if axes is None:
            axes = tuple(range(x.ndim - 1, -1, -1))
        else:
            axes = tuple(axes)
            axes = tuple(a if a >= 0 else a + x.ndim for a in axes)
            if len(axes) != x.ndim:
                raise ValueError(f"transpose axes length {len(axes)} must match ndim {x.ndim}")
            if set(axes) != set(range(x.ndim)):
                raise ValueError(f"transpose axes {axes} is not a valid permutation")
        inv_axes = tuple(np.argsort(axes))
        ctx.save_for_backward(inv_axes=inv_axes)
        return np.transpose(x, axes)

    @staticmethod
    def backward(ctx, g):
        return np.transpose(g, ctx.saved_data["inv_axes"])


class _Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x_shape=x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, g):
        return g.reshape(ctx.saved_data["x_shape"])


class _Slice(Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(x_shape=x.shape, x_dtype=x.dtype, idx=idx)
        return x[idx]

    @staticmethod
    def backward(ctx, g):
        x_shape = ctx.saved_data["x_shape"]
        x_dtype = ctx.saved_data["x_dtype"]
        idx = ctx.saved_data["idx"]
        full = np.zeros(x_shape, dtype=x_dtype)
        np.add.at(full, idx, g)  # handles duplicate indices
        return full


class _Concatenate(Function):
    @staticmethod
    def forward(ctx, axis, *tensors):
        ndim = tensors[0].ndim if tensors else 0
        axis_norm = axis if axis >= 0 else axis + ndim
        ctx.save_for_backward(axis=axis, splits=tuple(t.shape[axis] for t in tensors))
        return np.concatenate(tensors, axis=axis)

    @staticmethod
    def backward(ctx, g):
        axis = ctx.saved_data["axis"]
        splits = ctx.saved_data["splits"]
        return tuple(np.split(g, np.cumsum(splits[:-1]), axis=axis))


class _Tile(Function):
    @staticmethod
    def forward(ctx, x, reps):
        ctx.save_for_backward(x_shape=x.shape, reps=tuple(reps))
        return np.tile(x, reps)

    @staticmethod
    def backward(ctx, g):
        x_shape = tuple(ctx.saved_data["x_shape"])
        reps = tuple(ctx.saved_data["reps"])

        # Align reps length with x ndim
        nd_out = max(len(x_shape), len(reps))
        eff_x = (1,) * (nd_out - len(x_shape)) + x_shape
        reps_padded = (1,) * (nd_out - len(reps)) + reps

        # Sum over tiled copies, last axis first to keep indices stable
        for axis in reversed(range(nd_out)):
            r = reps_padded[axis]
            if r == 1:
                continue
            try:
                g = sum(np.split(g, r, axis=axis))
            except ValueError:
                # Fallback for non-even splits via reshape
                shape = list(g.shape)
                shape[axis] = eff_x[axis]
                shape.insert(axis + 1, r)
                g = g.reshape(shape).sum(axis=axis + 1)

        # Remove leading padded dims
        if g.shape != x_shape:
            try:
                g = g.reshape(x_shape)
            except ValueError:
                leading = len(g.shape) - len(x_shape)
                if leading > 0:
                    g = g.sum(axis=tuple(range(leading)))
                g = g.reshape(x_shape)
        return g


class _Repeat(Function):
    @staticmethod
    def forward(ctx, x, repeats, axis):
        axis_norm = axis if axis >= 0 else axis + x.ndim
        ctx.save_for_backward(x_shape=x.shape, repeats=repeats, axis=axis_norm)
        return np.repeat(x, repeats, axis=axis_norm)

    @staticmethod
    def backward(ctx, g):
        x_shape = ctx.saved_data["x_shape"]
        repeats = ctx.saved_data["repeats"]
        axis = ctx.saved_data["axis"]
        shape = list(g.shape)
        shape[axis] = x_shape[axis]
        shape.insert(axis + 1, repeats)
        return g.reshape(shape).sum(axis=axis + 1)


# ---------------------------------------------------------------------------
# Elementwise unary
# ---------------------------------------------------------------------------

class _Relu(Function):
    @staticmethod
    def forward(ctx, x):
        mask = x > 0
        ctx.save_for_backward(mask=mask)
        return np.maximum(0, x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.saved_data["mask"].astype(float)


class _Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        ctx.save_for_backward(s=s)
        return s

    @staticmethod
    def backward(ctx, g):
        s = ctx.saved_data["s"]
        return g * s * (1.0 - s)


class _Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        t = np.tanh(x)
        ctx.save_for_backward(t=t)
        return t

    @staticmethod
    def backward(ctx, g):
        t = ctx.saved_data["t"]
        return g * (1.0 - t ** 2)


class _Silu(Function):
    @staticmethod
    def forward(ctx, x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        ctx.save_for_backward(x=x, s=s)
        return x * s

    @staticmethod
    def backward(ctx, g):
        x, s = ctx.saved_data["x"], ctx.saved_data["s"]
        return g * (s + x * s * (1.0 - s))


class _Softmax(Function):
    @staticmethod
    def forward(ctx, x, axis):
        axis = -1 if axis is None else axis
        x_max = np.max(x, axis=axis, keepdims=True)
        exps = np.exp(x - x_max)
        s = exps / np.sum(exps, axis=axis, keepdims=True)
        ctx.save_for_backward(s=s, axis=axis)
        return s

    @staticmethod
    def backward(ctx, g):
        s, axis = ctx.saved_data["s"], ctx.saved_data["axis"]
        dot = np.sum(s * g, axis=axis, keepdims=True)
        return s * (g - dot)


class _Sqrt(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.sqrt(np.maximum(x, 0))
        ctx.save_for_backward(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        out = ctx.saved_data["out"]
        return g / (2.0 * np.maximum(out, 1e-15))


class _Exp(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.exp(x)
        ctx.save_for_backward(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        return g * ctx.saved_data["out"]


class _Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.log(np.maximum(x, 1e-15))

    @staticmethod
    def backward(ctx, g):
        return g / np.maximum(ctx.saved_data["x"], 1e-15)


class _Clip(Function):
    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x=x, a=a, b=b)
        return np.clip(x, a, b)

    @staticmethod
    def backward(ctx, g):
        x, a, b = ctx.saved_data["x"], ctx.saved_data["a"], ctx.saved_data["b"]
        return g * ((x >= a) & (x <= b)).astype(float)


class _Abs(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.abs(x)

    @staticmethod
    def backward(ctx, g):
        return g * np.sign(ctx.saved_data["x"])


class _Maximum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return np.maximum(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data["a"], ctx.saved_data["b"]
        # Distribute ties equally for unbiased gradient
        mask_a = (a > b).astype(float)
        mask_b = (b > a).astype(float)
        tie = (a == b).astype(float) * 0.5
        return g * (mask_a + tie), g * (mask_b + tie)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add(a, b):
    return _Add.apply(_ensure_tensor(a), _ensure_tensor(b))


def sub(a, b):
    return _Sub.apply(_ensure_tensor(a), _ensure_tensor(b))


def mul(a, b):
    return _Mul.apply(_ensure_tensor(a), _ensure_tensor(b))


def div(a, b):
    return _Div.apply(_ensure_tensor(a), _ensure_tensor(b))


def neg(x):
    return _Neg.apply(_ensure_tensor(x))


def _pow(x, power):
    if not isinstance(power, (int, float, np.integer, np.floating)):
        raise ValueError("_pow only supports constant scalar exponent")
    return _Pow.apply(_ensure_tensor(x), power)


def matmul(a, b):
    return _Matmul.apply(_ensure_tensor(a), _ensure_tensor(b))


def _sum(x, axis=None, keepdims=False):
    return _Sum.apply(_ensure_tensor(x), axis=axis, keepdims=keepdims)


def _mean(x, axis=None, keepdims=False):
    return _Mean.apply(_ensure_tensor(x), axis=axis, keepdims=keepdims)


def transpose(x, axes=None):
    return _Transpose.apply(_ensure_tensor(x), axes)


def reshape(x, shape):
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"reshape shape must be tuple/list, got {type(shape)}")
    return _Reshape.apply(_ensure_tensor(x), tuple(shape))


def slice_op(x, idx):
    return _Slice.apply(_ensure_tensor(x), idx)


def relu(x):
    return _Relu.apply(_ensure_tensor(x))


def sigmoid(x):
    return _Sigmoid.apply(_ensure_tensor(x))


def tanh(x):
    return _Tanh.apply(_ensure_tensor(x))


def silu(x):
    return _Silu.apply(_ensure_tensor(x))


def softmax(x, axis=-1):
    return _Softmax.apply(_ensure_tensor(x), axis=axis)


def sqrt(x):
    return _Sqrt.apply(_ensure_tensor(x))


def exp(x):
    return _Exp.apply(_ensure_tensor(x))


def log(x):
    return _Log.apply(_ensure_tensor(x))


def max_op(x, axis=None, keepdims=False):
    return _Max.apply(_ensure_tensor(x), axis=axis, keepdims=keepdims)


def concatenate(tensors, axis=0):
    tensors = [_ensure_tensor(t) for t in tensors]
    if not tensors:
        raise ValueError("concatenate requires at least one tensor")
    return _Concatenate.apply(axis, *tensors)


def tile(x, reps):
    reps = tuple(reps) if isinstance(reps, (tuple, list)) else (reps,)
    if any(not isinstance(r, int) or r <= 0 for r in reps):
        raise ValueError(f"tile reps must be positive ints, got {reps}")
    return _Tile.apply(_ensure_tensor(x), reps)


def clip(x, a, b):
    if a is not None and b is not None and np.asarray(a).size == 1 and np.asarray(b).size == 1:
        if float(np.asarray(a)) > float(np.asarray(b)):
            raise ValueError(f"clip: a ({a}) must be <= b ({b})")
    return _Clip.apply(_ensure_tensor(x), a, b)


def abs_op(x):
    return _Abs.apply(_ensure_tensor(x))


def repeat_op(x, repeats, axis):
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError(f"repeat_op repeats must be positive int, got {repeats}")
    return _Repeat.apply(_ensure_tensor(x), repeats, axis)


def maximum(a, b):
    return _Maximum.apply(_ensure_tensor(a), _ensure_tensor(b))


def var(x, axis=None, keepdims=False):
    x_t = _ensure_tensor(x)
    mean = _mean(x_t, axis=axis, keepdims=True)
    diff = x_t - mean
    return _mean(diff ** 2, axis=axis, keepdims=keepdims)
