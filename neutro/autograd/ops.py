import numpy as np
from .tensor import Tensor
from .function import Function


def _ensure_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


class _Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a + b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data['a'], ctx.saved_data['b']
        return g, g


class _Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
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
        a, b = ctx.saved_data['a'], ctx.saved_data['b']
        return g * b, g * a


class _Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a / b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data['a'], ctx.saved_data['b']
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
        x, k = ctx.saved_data['x'], ctx.saved_data['k']
        return k * (x ** (k - 1)) * g


class _Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a @ b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data['a'], ctx.saved_data['b']
        ga = g @ b.T if g.ndim == 2 and b.ndim == 2 else g @ np.swapaxes(b, -1, -2)
        gb = a.T @ g if g.ndim == 2 and a.ndim == 2 else np.swapaxes(a, -1, -2) @ g
        return ga, gb


class _Sum(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        ctx.save_for_backward(x_shape=x.shape, axis=axis, keepdims=keepdims)
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x_shape, axis, keepdims = ctx.saved_data['x_shape'], ctx.saved_data['axis'], ctx.saved_data['keepdims']
        if axis is None:
            gx = g * np.ones(x_shape, dtype=float)
        else:
            gx = np.expand_dims(g, axis=axis) if not keepdims else g
            gx = gx * np.ones(x_shape, dtype=float)
        return gx.reshape(x_shape)


class _Mean(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        if axis is None:
            n = x.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            n = int(np.prod([x.shape[d] for d in axes]))
        ctx.save_for_backward(x_shape=x.shape, axis=axis, keepdims=keepdims, n=n)
        return np.mean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x_shape, axis, keepdims, n = ctx.saved_data['x_shape'], ctx.saved_data['axis'], ctx.saved_data['keepdims'], ctx.saved_data['n']
        if axis is None:
            gx = g * np.ones(x_shape, dtype=float) / n
        else:
            gx = np.expand_dims(g, axis=axis) if not keepdims else g
            gx = gx * np.ones(x_shape, dtype=float) / n
        return gx.reshape(x_shape)


class _Transpose(Function):
    @staticmethod
    def forward(ctx, x, axes):
        if axes is None:
            axes = tuple(range(x.ndim - 1, -1, -1))
        ctx.save_for_backward(inv_axes=tuple(np.argsort(axes)))
        return np.transpose(x, axes)

    @staticmethod
    def backward(ctx, g):
        return np.transpose(g, ctx.saved_data['inv_axes'])


class _Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x_shape=x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, g):
        return g.reshape(ctx.saved_data['x_shape'])


class _Slice(Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(x=x, idx=idx)
        return x[idx]

    @staticmethod
    def backward(ctx, g):
        full = np.zeros_like(ctx.saved_data['x'])
        full[ctx.saved_data['idx']] += g
        return full


class _Relu(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.maximum(0, x)

    @staticmethod
    def backward(ctx, g):
        return g * (ctx.saved_data['x'] > 0).astype(float)


class _Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        ctx.save_for_backward(s=s.copy())
        return s.copy()

    @staticmethod
    def backward(ctx, g):
        s = ctx.saved_data['s']
        return g * s * (1.0 - s)


class _Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.tanh(x)

    @staticmethod
    def backward(ctx, g):
        t = np.tanh(ctx.saved_data['x'])
        return g * (1.0 - t ** 2)


class _Silu(Function):
    @staticmethod
    def forward(ctx, x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        ctx.save_for_backward(x=x, s=s.copy())
        return x * s

    @staticmethod
    def backward(ctx, g):
        x, s = ctx.saved_data['x'], ctx.saved_data['s']
        return g * (s + x * s * (1.0 - s))


class _Softmax(Function):
    @staticmethod
    def forward(ctx, x, axis):
        x_max = np.max(x, axis=axis, keepdims=True)
        exps = np.exp(x - x_max)
        s = exps / np.sum(exps, axis=axis, keepdims=True)
        ctx.save_for_backward(s=s.copy(), axis=axis)
        return s.copy()

    @staticmethod
    def backward(ctx, g):
        s, axis = ctx.saved_data['s'], ctx.saved_data['axis']
        dot = np.sum(s * g, axis=axis, keepdims=True)
        return s * (g - dot)


class _Sqrt(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.sqrt(x)

    @staticmethod
    def backward(ctx, g):
        return g / (2.0 * np.sqrt(ctx.saved_data['x']) + 1e-15)


class _Exp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.exp(x)

    @staticmethod
    def backward(ctx, g):
        return g * np.exp(ctx.saved_data['x'])


class _Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.log(x + 1e-15)

    @staticmethod
    def backward(ctx, g):
        return g / (ctx.saved_data['x'] + 1e-15)


class _Max(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        ctx.save_for_backward(x=x, axis=axis, keepdims=keepdims)
        return np.max(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x, axis, keepdims = ctx.saved_data['x'], ctx.saved_data['axis'], ctx.saved_data['keepdims']
        mask = (x == np.max(x, axis=axis, keepdims=True)).astype(float)
        s = mask.sum(axis=axis, keepdims=True) if axis is not None else mask.sum()
        s = np.clip(s, 1e-15, None)
        mask = mask / s
        if axis is None:
            gx = g * mask
        else:
            gx = np.expand_dims(g, axis=axis) if not keepdims else g
            gx = gx * mask
        return gx.reshape(x.shape)


class _Concatenate(Function):
    @staticmethod
    def forward(ctx, axis, *tensors):
        ctx.save_for_backward(axis=axis, splits=tuple(t.shape[axis] for t in tensors))
        return np.concatenate(tensors, axis=axis)

    @staticmethod
    def backward(ctx, g):
        axis = ctx.saved_data['axis']
        splits = ctx.saved_data['splits']
        return tuple(np.split(g, np.cumsum(splits[:-1]), axis=axis))


class _Tile(Function):
    @staticmethod
    def forward(ctx, x, reps):
        ctx.save_for_backward(x_shape=x.shape, reps=reps)
        return np.tile(x, reps)

    @staticmethod
    def backward(ctx, g):
        x_shape, reps = ctx.saved_data['x_shape'], ctx.saved_data['reps']
        for ax, r in enumerate(reps):
            if r > 1:
                g = np.add.reduceat(g, np.arange(0, g.shape[ax], r), axis=ax)
        return g.reshape(x_shape)


class _Clip(Function):
    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x=x, a=a, b=b)
        return np.clip(x, a, b)

    @staticmethod
    def backward(ctx, g):
        x, a, b = ctx.saved_data['x'], ctx.saved_data['a'], ctx.saved_data['b']
        return g * ((x >= a) & (x <= b)).astype(float)


class _Abs(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.abs(x)

    @staticmethod
    def backward(ctx, g):
        return g * np.sign(ctx.saved_data['x'] + 1e-15)


class _Repeat(Function):
    @staticmethod
    def forward(ctx, x, repeats, axis):
        ctx.save_for_backward(x_shape=x.shape, repeats=repeats, axis=axis)
        return np.repeat(x, repeats, axis=axis)

    @staticmethod
    def backward(ctx, g):
        x_shape, repeats, axis = ctx.saved_data['x_shape'], ctx.saved_data['repeats'], ctx.saved_data['axis']
        shape = list(g.shape)
        shape.insert(axis + 1, repeats)
        shape[axis] = x_shape[axis]
        return g.reshape(shape).sum(axis=axis + 1)


class _Maximum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return np.maximum(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data['a'], ctx.saved_data['b']
        mask = (a >= b).astype(float)
        return g * mask, g * (1.0 - mask)


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
    if not isinstance(power, (int, float)):
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
    return _Reshape.apply(_ensure_tensor(x), shape)


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
    return _Concatenate.apply(axis, *tensors)


def tile(x, reps):
    reps = tuple(reps) if isinstance(reps, (tuple, list)) else (reps,)
    return _Tile.apply(_ensure_tensor(x), reps)


def clip(x, a, b):
    return _Clip.apply(_ensure_tensor(x), a, b)


def abs_op(x):
    return _Abs.apply(_ensure_tensor(x))


def repeat_op(x, repeats, axis):
    return _Repeat.apply(_ensure_tensor(x), repeats, axis)


def maximum(a, b):
    return _Maximum.apply(_ensure_tensor(a), _ensure_tensor(b))


def var(x, axis=None, keepdims=False):
    mean = _mean(x, axis=axis, keepdims=True)
    diff = x - mean
    return _mean(diff ** 2, axis=axis, keepdims=keepdims)
