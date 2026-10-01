import numpy as np
from .tensor import Tensor
from .function import Function


def _ensure_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x))


# ------------------------------------------------------------------
# Helpers for axis handling
# ------------------------------------------------------------------
def _normalize_axis(axis, ndim):
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    # normalize negative
    axis = tuple(a if a >= 0 else a + ndim for a in axis)
    # validate
    for a in axis:
        if not 0 <= a < ndim:
            raise ValueError(f"axis {a} out of bounds for ndim {ndim}")
    # unique sorted
    if len(set(axis)) != len(axis):
        raise ValueError(f"repeated axis in {axis}")
    return axis


def _expand_grad_to_shape(g, x_shape, axis, keepdims):
    """
    Broadcast gradient g back to x_shape for reduction ops (sum/mean/max).
    g is the upstream gradient with shape equal to reduced shape.
    """
    x_shape = tuple(x_shape)
    if axis is None:
        # g is scalar (0-d) or shape () -> broadcast to x_shape
        return np.broadcast_to(np.asarray(g), x_shape).copy()

    axes = _normalize_axis(axis, len(x_shape))
    if keepdims:
        # g already has ndim == len(x_shape) with 1s at reduced axes
        return np.broadcast_to(np.asarray(g), x_shape).copy()
    else:
        # Need to insert singleton dims at reduced axes
        g_arr = np.asarray(g)
        # Build expanded shape where reduced axes are 1
        # g_arr ndim == len(x_shape) - len(axes)
        expanded_shape = []
        g_idx = 0
        axes_set = set(axes)
        for i in range(len(x_shape)):
            if i in axes_set:
                expanded_shape.append(1)
            else:
                expanded_shape.append(g_arr.shape[g_idx])
                g_idx += 1
        g_reshaped = g_arr.reshape(expanded_shape)
        return np.broadcast_to(g_reshaped, x_shape).copy()


def _swap_last_two(x):
    if x.ndim < 2:
        return x
    return np.swapaxes(x, -1, -2)


class _Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # No need to save a,b for backward (gradient is just upstream)
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
        # Avoid division by zero issues; b is numpy array
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
        # Handle k == 0 separately to avoid x**( -1 ) for x==0
        if k == 0:
            return np.zeros_like(x, dtype=float) * g
        return k * (x ** (k - 1)) * g


class _Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return a @ b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data['a'], ctx.saved_data['b']
        # Handle vector cases explicitly
        if a.ndim == 1 and b.ndim == 1:
            # dot product: a (K,) @ b (K,) -> scalar, g scalar
            # ga = g * b, gb = g * a
            return g * b, g * a
        elif a.ndim == 1 and b.ndim >= 2:
            # a (K,) @ b (..., K, N) -> (..., N)
            # For simplicity promote a to (1, K) and use generic formula, then squeeze
            # But generic _swap_last_two path also works if we promote g?
            # Use einsum for correctness: ga = b @ g? For non-batched: b (K,N) @ g (N,) -> (K,)
            # So ga = b @ g, gb = outer(a, g)
            # For batched, use tensordot / matmul with broadcasting
            # Fallback to generic with expanded dims
            # Promote a
            # We'll compute ga via matmul(b, g_expanded)
            # Simpler: use np.matmul with swapped axes after expanding a
            a_exp = a[np.newaxis, :]  # (1, K)
            # g may be batched: need to handle batch dims
            # If b is 2D and g is 1D, simple:
            if b.ndim == 2 and g.ndim == 1:
                ga = b @ g  # (K,)
                gb = np.outer(a, g)  # (K, N)
                return ga, gb
            # For higher dims, fallback to generic with swapaxes after promoting
            # Generic fallback: treat as batched matmul with leading batch
            ga = np.matmul(g, _swap_last_two(b)) if g.ndim >= 1 and b.ndim >= 2 else g * b
            # ga currently shape (..., K) but may have extra leading 1 from a promotion
            # Need to squeeze the leading 1 if added
            if ga.ndim > 1 and ga.shape[0] == 1 and a.ndim == 1:
                # heuristic squeeze first dim if it was added
                # Actually ga from g @ b.T where b shape (K,N) -> g shape (N,) -> g @ b.T = (K,) without leading
                pass
            gb = np.matmul(a_exp.T, g[np.newaxis, :] if g.ndim == 1 else g) if g.ndim >= 1 else np.outer(a, g)
            # gb shape currently (K, N) or (..., K, N) - may need to handle batch
            return ga, gb
        elif b.ndim == 1 and a.ndim >= 2:
            # a (..., M, K) @ b (K,) -> (..., M)
            if a.ndim == 2 and g.ndim == 1:
                ga = np.outer(g, b)  # (M, K)
                gb = a.T @ g  # (K,)
                return ga, gb
            # Batched fallback
            ga = np.matmul(g[..., None], b[None, :]) if g.ndim >= 1 else g * b
            # g[..., None] shape (..., M, 1) @ b[None,:] (1, K) -> (..., M, K) ?
            # Use generic swap method
            ga_generic = np.matmul(g[..., None], _swap_last_two(b)[None, :]) if False else None
            # For robustness use swap_last_two generic
            # Recompute using generic but with vector handling
            ga = g[..., None] * b if g.ndim >= 1 else g * b  # outer
            gb = np.matmul(_swap_last_two(a), g[..., None]).squeeze(-1) if a.ndim >=2 else g * a
            return ga, gb

        # Generic case: both ndim >=2 (includes batched)
        ga = np.matmul(g, _swap_last_two(b))
        gb = np.matmul(_swap_last_two(a), g)
        return ga, gb


class _Sum(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        ctx.save_for_backward(x_shape=x.shape, axis=axis, keepdims=keepdims)
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x_shape, axis, keepdims = ctx.saved_data['x_shape'], ctx.saved_data['axis'], ctx.saved_data['keepdims']
        return _expand_grad_to_shape(g, x_shape, axis, keepdims)


class _Mean(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        if axis is None:
            n = x.size
        else:
            axes = _normalize_axis(axis, x.ndim)
            n = int(np.prod([x.shape[d] for d in axes]))
            # Save normalized axis for backward
            axis = axes if isinstance(axis, (tuple, list)) else axis
        ctx.save_for_backward(x_shape=x.shape, axis=axis, keepdims=keepdims, n=n)
        return np.mean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x_shape, axis, keepdims, n = ctx.saved_data['x_shape'], ctx.saved_data['axis'], ctx.saved_data['keepdims'], ctx.saved_data['n']
        gx = _expand_grad_to_shape(g, x_shape, axis, keepdims)
        return gx / n


class _Transpose(Function):
    @staticmethod
    def forward(ctx, x, axes):
        if axes is None:
            axes = tuple(range(x.ndim - 1, -1, -1))
        else:
            axes = tuple(axes)
            # Normalize negative axes
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
        # Save only shape and idx, not full x (memory efficient)
        ctx.save_for_backward(x_shape=x.shape, x_dtype=x.dtype, idx=idx)
        return x[idx]

    @staticmethod
    def backward(ctx, g):
        x_shape = ctx.saved_data['x_shape']
        x_dtype = ctx.saved_data['x_dtype']
        idx = ctx.saved_data['idx']
        full = np.zeros(x_shape, dtype=x_dtype)
        # Use np.add.at for correct accumulation with duplicate indices (fancy indexing)
        np.add.at(full, idx, g)
        return full


class _Relu(Function):
    @staticmethod
    def forward(ctx, x):
        mask = x > 0
        ctx.save_for_backward(mask=mask)
        return np.maximum(0, x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.saved_data['mask'].astype(float)


class _Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        # Save s for backward; keep copy to avoid aliasing with output
        ctx.save_for_backward(s=s)
        return s

    @staticmethod
    def backward(ctx, g):
        s = ctx.saved_data['s']
        return g * s * (1.0 - s)


class _Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        t = np.tanh(x)
        ctx.save_for_backward(t=t)
        return t

    @staticmethod
    def backward(ctx, g):
        t = ctx.saved_data['t']
        return g * (1.0 - t ** 2)


class _Silu(Function):
    @staticmethod
    def forward(ctx, x):
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        ctx.save_for_backward(x=x, s=s)
        return x * s

    @staticmethod
    def backward(ctx, g):
        x, s = ctx.saved_data['x'], ctx.saved_data['s']
        return g * (s + x * s * (1.0 - s))


class _Softmax(Function):
    @staticmethod
    def forward(ctx, x, axis):
        # Normalize axis
        ndim = x.ndim
        if axis is None:
            axis = -1
        axis_norm = axis if axis >= 0 else axis + ndim
        x_max = np.max(x, axis=axis, keepdims=True)
        exps = np.exp(x - x_max)
        s = exps / np.sum(exps, axis=axis, keepdims=True)
        ctx.save_for_backward(s=s, axis=axis)
        return s

    @staticmethod
    def backward(ctx, g):
        s, axis = ctx.saved_data['s'], ctx.saved_data['axis']
        dot = np.sum(s * g, axis=axis, keepdims=True)
        return s * (g - dot)


class _Sqrt(Function):
    @staticmethod
    def forward(ctx, x):
        # Clamp negative to 0 for stability? Assume x >=0
        out = np.sqrt(np.maximum(x, 0))
        ctx.save_for_backward(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        out = ctx.saved_data['out']
        # grad = g / (2 * sqrt(x)) = g / (2*out); handle out==0 with eps
        return g / (2.0 * np.maximum(out, 1e-15))


class _Exp(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.exp(x)
        ctx.save_for_backward(out=out)
        return out

    @staticmethod
    def backward(ctx, g):
        return g * ctx.saved_data['out']


class _Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.log(np.maximum(x, 1e-15))

    @staticmethod
    def backward(ctx, g):
        return g / np.maximum(ctx.saved_data['x'], 1e-15)


class _Max(Function):
    @staticmethod
    def forward(ctx, x, axis, keepdims):
        ctx.save_for_backward(x=x, axis=axis, keepdims=keepdims)
        return np.max(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        x, axis, keepdims = ctx.saved_data['x'], ctx.saved_data['axis'], ctx.saved_data['keepdims']
        # Create mask for max positions; handle ties by dividing equally
        x_max = np.max(x, axis=axis, keepdims=True)
        mask = (x == x_max).astype(float)
        if axis is None:
            s = mask.sum()
            s = np.clip(s, 1e-15, None)
            mask = mask / s
            gx = g * mask  # g is scalar
        else:
            axes = _normalize_axis(axis, x.ndim)
            # Sum over reduced axes with keepdims True to get count per slice
            s = mask.sum(axis=axis, keepdims=True)
            s = np.clip(s, 1e-15, None)
            mask = mask / s
            # Expand g to x shape
            gx_expanded = _expand_grad_to_shape(g, x.shape, axis, keepdims)
            gx = gx_expanded * mask
        return gx


class _Concatenate(Function):
    @staticmethod
    def forward(ctx, axis, *tensors):
        # Normalize axis
        ndim = tensors[0].ndim if tensors else 0
        axis_norm = axis if axis >= 0 else axis + ndim
        if not 0 <= axis_norm < ndim:
            # Allow axis == ndim for 1D? but numpy allows. Validate via first tensor
            pass
        ctx.save_for_backward(axis=axis, axis_norm=axis_norm, splits=tuple(t.shape[axis] for t in tensors), ndim=ndim)
        return np.concatenate(tensors, axis=axis)

    @staticmethod
    def backward(ctx, g):
        axis = ctx.saved_data['axis']
        splits = ctx.saved_data['splits']
        # Use normalized axis for split but numpy accepts negative too; keep original
        return tuple(np.split(g, np.cumsum(splits[:-1]), axis=axis))


class _Tile(Function):
    @staticmethod
    def forward(ctx, x, reps):
        ctx.save_for_backward(x_shape=x.shape, reps=tuple(reps))
        return np.tile(x, reps)

    @staticmethod
    def backward(ctx, g):
        x_shape, reps = ctx.saved_data['x_shape'], ctx.saved_data['reps']
        # Robust handling for reps length mismatch with ndim
        x_shape = tuple(x_shape)
        reps = tuple(reps)
        nd_out = max(len(x_shape), len(reps))
        eff_x = (1,) * (nd_out - len(x_shape)) + x_shape
        reps_padded = (1,) * (nd_out - len(reps)) + reps
        # Iteratively sum over tiled repetitions
        # Process from last axis to first to keep axis indices stable
        for axis in reversed(range(nd_out)):
            r = reps_padded[axis]
            if r == 1:
                continue
            # g.shape[axis] should be eff_x[axis] * r
            # Split and sum
            # Use split; handle case where g.shape[axis] not divisible due to rounding? Should be exact.
            try:
                parts = np.split(g, r, axis=axis)
            except ValueError:
                # Fallback: reshape then sum
                # e.g., g.shape[axis] = eff_x[axis]*r, split via reshape
                shape = list(g.shape)
                shape[axis] = eff_x[axis]
                shape.insert(axis + 1, r)
                g_reshaped = g.reshape(shape)
                g = g_reshaped.sum(axis=axis + 1)
                continue
            g = sum(parts)
        # Now g shape == eff_x
        if g.shape != x_shape:
            # Remove leading padded dims by squeezing/reshaping
            try:
                g = g.reshape(x_shape)
            except ValueError:
                # Sum over leading dims that were added
                leading = len(eff_x) - len(x_shape)
                if leading > 0:
                    g = g.sum(axis=tuple(range(leading)))
                    g = g.reshape(x_shape)
        return g


class _Clip(Function):
    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x=x, a=a, b=b)
        return np.clip(x, a, b)

    @staticmethod
    def backward(ctx, g):
        x, a, b = ctx.saved_data['x'], ctx.saved_data['a'], ctx.saved_data['b']
        # Gradient is zero where clipped (outside [a,b]), 1 inside inclusive
        # Handle scalar a,b vs array
        return g * ((x >= a) & (x <= b)).astype(float)


class _Abs(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return np.abs(x)

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_data['x']
        # Subgradient at 0 is 0
        sign = np.sign(x)
        # np.sign(0) == 0 already, good
        return g * sign


class _Repeat(Function):
    @staticmethod
    def forward(ctx, x, repeats, axis):
        # Normalize axis
        ndim = x.ndim
        axis_norm = axis if axis >= 0 else axis + ndim
        ctx.save_for_backward(x_shape=x.shape, repeats=repeats, axis=axis_norm)
        return np.repeat(x, repeats, axis=axis_norm)

    @staticmethod
    def backward(ctx, g):
        x_shape, repeats, axis = ctx.saved_data['x_shape'], ctx.saved_data['repeats'], ctx.saved_data['axis']
        # g shape along axis is x_shape[axis] * repeats
        # Reshape to (..., x_shape[axis], repeats, ...) and sum over repeats
        # Implementation: reshape to insert repeats dim
        # For robustness handle negative axis already normalized
        shape = list(g.shape)
        # Insert repeats dimension after axis
        # But g.shape[axis] = x_shape[axis] * repeats
        # So we reshape: shape[axis] = x_shape[axis], insert repeats
        shape[axis] = x_shape[axis]
        shape.insert(axis + 1, repeats)
        return g.reshape(shape).sum(axis=axis + 1)


class _Maximum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return np.maximum(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_data['a'], ctx.saved_data['b']
        # Tie case: gradient goes to both? Use standard where a>=b gets grad
        # For robustness distribute equally on ties to avoid bias
        mask_a = (a > b).astype(float)
        mask_b = (b > a).astype(float)
        tie = (a == b).astype(float)
        # Split ties equally
        mask_a += tie * 0.5
        mask_b += tie * 0.5
        return g * mask_a, g * mask_b


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
    # Validate shape
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
    diff = x_t - mean  # uses sub with broadcasting
    return _mean(diff ** 2, axis=axis, keepdims=keepdims)
