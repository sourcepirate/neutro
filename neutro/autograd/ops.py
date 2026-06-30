import numpy as np
from .tensor import Tensor
from .tape import get_active_tape
from .utils import broadcast_backward


def _ensure_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def add(a, b):
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = a.data + b.data
    result = Tensor(out)
    tape = get_active_tape()
    if tape and (id(a) in tape._watched or id(b) in tape._watched):
        a_shape, b_shape = a.shape, b.shape
        def bw(g):
            return [broadcast_backward(g, a_shape),
                    broadcast_backward(g, b_shape)]
        tape._record_op([a, b], result, bw, 'add')
    return result


def sub(a, b):
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = a.data - b.data
    result = Tensor(out)
    tape = get_active_tape()
    if tape and (id(a) in tape._watched or id(b) in tape._watched):
        a_shape, b_shape = a.shape, b.shape
        def bw(g):
            return [broadcast_backward(g, a_shape),
                    broadcast_backward(-g, b_shape)]
        tape._record_op([a, b], result, bw, 'sub')
    return result


def mul(a, b):
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = a.data * b.data
    result = Tensor(out)
    tape = get_active_tape()
    if tape and (id(a) in tape._watched or id(b) in tape._watched):
        a_shape, b_shape = a.shape, b.shape
        def bw(g):
            ga = broadcast_backward(g * b.data, a_shape) if id(a) in tape._watched else None
            gb = broadcast_backward(g * a.data, b_shape) if id(b) in tape._watched else None
            return [ga, gb]
        tape._record_op([a, b], result, bw, 'mul')
    return result


def div(a, b):
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = a.data / b.data
    result = Tensor(out)
    tape = get_active_tape()
    if tape and (id(a) in tape._watched or id(b) in tape._watched):
        a_shape, b_shape = a.shape, b.shape
        def bw(g):
            ga = broadcast_backward(g / b.data, a_shape) if id(a) in tape._watched else None
            gb = broadcast_backward(-g * a.data / (b.data ** 2), b_shape) if id(b) in tape._watched else None
            return [ga, gb]
        tape._record_op([a, b], result, bw, 'div')
    return result


def neg(x):
    x = _ensure_tensor(x)
    out = -x.data
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            return [-g]
        tape._record_op([x], result, bw, 'neg')
    return result


def _pow(x, power):
    x = _ensure_tensor(x)
    if not isinstance(power, (int, float)):
        raise ValueError("_pow only supports constant scalar exponent")
    out = x.data ** power
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        k = power
        def bw(g):
            return [(k * (x.data ** (k - 1)) * g)]
        tape._record_op([x], result, bw, 'pow')
    return result


def matmul(a, b):
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = a.data @ b.data
    result = Tensor(out)
    tape = get_active_tape()
    if tape and (id(a) in tape._watched or id(b) in tape._watched):
        a_shape, b_shape = a.shape, b.shape
        def bw(g):
            ga = None
            if id(a) in tape._watched:
                if g.ndim == 2 and b.data.ndim == 2:
                    ga = g @ b.data.T
                else:
                    ga = g @ np.swapaxes(b.data, -1, -2)
                ga = broadcast_backward(ga, a_shape)
            gb = None
            if id(b) in tape._watched:
                if g.ndim == 2 and a.data.ndim == 2:
                    gb = a.data.T @ g
                else:
                    gb = np.swapaxes(a.data, -1, -2) @ g
                gb = broadcast_backward(gb, b_shape)
            return [ga, gb]
        tape._record_op([a, b], result, bw, 'matmul')
    return result


def _sum(x, axis=None, keepdims=False):
    x = _ensure_tensor(x)
    out = np.sum(x.data, axis=axis, keepdims=keepdims)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        x_shape = x.shape
        ax = axis
        def bw(g):
            if ax is None:
                gx = g * np.ones(x_shape, dtype=float)
            else:
                gx = np.expand_dims(g, axis=ax) if not keepdims else g
                gx = gx * np.ones(x_shape, dtype=float)
            return [gx.reshape(x_shape)]
        tape._record_op([x], result, bw, 'sum')
    return result


def _mean(x, axis=None, keepdims=False):
    x = _ensure_tensor(x)
    out = np.mean(x.data, axis=axis, keepdims=keepdims)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        x_shape = x.shape
        ax = axis
        if ax is None:
            n = x.data.size
        else:
            ax_t = (ax,) if isinstance(ax, int) else ax
            n = int(np.prod([x_shape[d] for d in ax_t]))
        def bw(g):
            if ax is None:
                gx = g * np.ones(x_shape, dtype=float) / n
            else:
                gx = np.expand_dims(g, axis=ax) if not keepdims else g
                gx = gx * np.ones(x_shape, dtype=float) / n
            return [gx.reshape(x_shape)]
        tape._record_op([x], result, bw, 'mean')
    return result


def transpose(x, axes=None):
    x = _ensure_tensor(x)
    if axes is None:
        axes = tuple(range(x.ndim - 1, -1, -1))
    out = np.transpose(x.data, axes)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        inv_axes = tuple(np.argsort(axes))
        def bw(g):
            return [np.transpose(g, inv_axes)]
        tape._record_op([x], result, bw, 'transpose')
    return result


def reshape(x, shape):
    x = _ensure_tensor(x)
    out = x.data.reshape(shape)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        x_shape = x.shape
        def bw(g):
            return [g.reshape(x_shape)]
        tape._record_op([x], result, bw, 'reshape')
    return result


def slice_op(x, idx):
    x = _ensure_tensor(x)
    out = x.data[idx]
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            full = np.zeros_like(x.data)
            full[idx] += g
            return [full]
        tape._record_op([x], result, bw, 'slice')
    return result


def relu(x):
    x = _ensure_tensor(x)
    out = np.maximum(0, x.data)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            return [g * (x.data > 0).astype(float)]
        tape._record_op([x], result, bw, 'relu')
    return result


def sigmoid(x):
    x = _ensure_tensor(x)
    s = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
    out = s.copy()
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        s_cache = s.copy()
        def bw(g):
            return [g * s_cache * (1.0 - s_cache)]
        tape._record_op([x], result, bw, 'sigmoid')
    return result


def tanh(x):
    x = _ensure_tensor(x)
    out = np.tanh(x.data)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            t = np.tanh(x.data)
            return [g * (1.0 - t ** 2)]
        tape._record_op([x], result, bw, 'tanh')
    return result


def silu(x):
    x = _ensure_tensor(x)
    s = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
    out = x.data * s
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        s_cache = s.copy()
        def bw(g):
            ds = s_cache * (1.0 - s_cache)
            return [g * (s_cache + x.data * ds)]
        tape._record_op([x], result, bw, 'silu')
    return result


def softmax(x, axis=-1):
    x = _ensure_tensor(x)
    x_max = np.max(x.data, axis=axis, keepdims=True)
    exps = np.exp(x.data - x_max)
    s = exps / np.sum(exps, axis=axis, keepdims=True)
    out = s.copy()
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        s_cache = s.copy()
        def bw(g):
            dot = np.sum(s_cache * g, axis=axis, keepdims=True)
            return [s_cache * (g - dot)]
        tape._record_op([x], result, bw, 'softmax')
    return result


def sqrt(x):
    x = _ensure_tensor(x)
    out = np.sqrt(x.data)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            return [g / (2.0 * np.sqrt(x.data) + 1e-15)]
        tape._record_op([x], result, bw, 'sqrt')
    return result


def exp(x):
    x = _ensure_tensor(x)
    out = np.exp(x.data)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            return [g * np.exp(x.data)]
        tape._record_op([x], result, bw, 'exp')
    return result


def log(x):
    x = _ensure_tensor(x)
    out = np.log(x.data + 1e-15)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            return [g / (x.data + 1e-15)]
        tape._record_op([x], result, bw, 'log')
    return result


def max_op(x, axis=None, keepdims=False):
    x = _ensure_tensor(x)
    out = np.max(x.data, axis=axis, keepdims=keepdims)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        x_shape = x.shape
        ax = axis
        def bw(g):
            mask = (x.data == np.max(x.data, axis=ax, keepdims=True)).astype(float)
            s = mask.sum(axis=ax, keepdims=True) if ax is not None else mask.sum()
            s = np.clip(s, 1e-15, None)
            mask = mask / s
            if ax is None:
                gx = g * mask
            else:
                gx = np.expand_dims(g, axis=ax) if not keepdims else g
                gx = gx * mask
            return [gx.reshape(x_shape)]
        tape._record_op([x], result, bw, 'max')
    return result


def concatenate(tensors, axis=0):
    tensors = [_ensure_tensor(t) for t in tensors]
    out = np.concatenate([t.data for t in tensors], axis=axis)
    result = Tensor(out)
    tape = get_active_tape()
    if tape:
        splits = [t.shape[axis] for t in tensors]
        def bw(g):
            grads = np.split(g, np.cumsum(splits[:-1]), axis=axis)
            return [g if id(t) in tape._watched else None
                    for g, t in zip(grads, tensors)]
        tape._record_op(tensors, result, bw, 'concatenate')
    return result


def tile(x, reps):
    x = _ensure_tensor(x)
    out = np.tile(x.data, reps)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        x_shape = x.shape
        reps_t = tuple(reps) if isinstance(reps, (tuple, list)) else (reps,)
        def bw(g):
            for ax, r in enumerate(reps_t):
                if r > 1:
                    g = np.add.reduceat(g, np.arange(0, g.shape[ax], x_shape[ax]), axis=ax)
            return [g.reshape(x_shape)]
        tape._record_op([x], result, bw, 'tile')
    return result


def clip(x, a, b):
    x = _ensure_tensor(x)
    out = np.clip(x.data, a, b)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            mask = ((x.data >= a) & (x.data <= b)).astype(float)
            return [g * mask]
        tape._record_op([x], result, bw, 'clip')
    return result


def abs_op(x):
    x = _ensure_tensor(x)
    out = np.abs(x.data)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        def bw(g):
            return [g * np.sign(x.data + 1e-15)]
        tape._record_op([x], result, bw, 'abs')
    return result


def var(x, axis=None, keepdims=False):
    mean = _mean(x, axis=axis, keepdims=True)
    diff = x - mean
    return _mean(diff ** 2, axis=axis, keepdims=keepdims)


def repeat_op(x, repeats, axis):
    x = _ensure_tensor(x)
    out = np.repeat(x.data, repeats, axis=axis)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and id(x) in tape._watched:
        x_shape = x.shape
        def bw(g):
            shape = list(g.shape)
            shape.insert(axis + 1, repeats)
            shape[axis] = x_shape[axis]
            return [g.reshape(shape).sum(axis=axis + 1)]
        tape._record_op([x], result, bw, 'repeat')
    return result


def maximum(a, b):
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    out = np.maximum(a.data, b.data)
    result = Tensor(out)
    tape = get_active_tape()
    if tape and (id(a) in tape._watched or id(b) in tape._watched):
        a_shape, b_shape = a.shape, b.shape
        def bw(g):
            mask = (a.data >= b.data).astype(float)
            ga = broadcast_backward(g * mask, a_shape) if id(a) in tape._watched else None
            gb = broadcast_backward(g * (1.0 - mask), b_shape) if id(b) in tape._watched else None
            return [ga, gb]
        tape._record_op([a, b], result, bw, 'maximum')
    return result
