"""Utilities for autograd gradient handling."""

import numpy as np


def broadcast_backward(grad, original_shape):
    """Reduce `grad` to `original_shape` by summing over broadcasted axes.

    Handles:
      - scalar originals -> sum to 0-d
      - leading broadcast dims -> sum over them
      - dims where original == 1 -> sum with keepdims
    """
    grad = np.asarray(grad)

    original_shape = tuple(original_shape)
    if grad.shape == original_shape:
        return grad

    # Scalar: all dims were broadcast.
    if len(original_shape) == 0:
        return grad.sum().reshape(original_shape)

    # Leading dimensions added by broadcasting.
    ndim_diff = grad.ndim - len(original_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    # Dimensions where original == 1 were broadcast.
    for axis, (g_dim, o_dim) in enumerate(zip(grad.shape, original_shape)):
        if o_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    if grad.shape != original_shape:
        try:
            grad = grad.reshape(original_shape)
        except ValueError as exc:
            raise ValueError(
                f"broadcast_backward: grad shape {grad.shape} "
                f"cannot be reduced to {original_shape}"
            ) from exc

    return grad


def numeric_grad(func, x, eps=1e-6):
    """Central-difference numeric gradient for testing.

    Args:
        func: callable ``(array) -> scalar``.
        x: array to differentiate.
        eps: finite-difference step.
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)

    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        plus = np.asarray(func(x)).sum()
        x[idx] = orig - eps
        minus = np.asarray(func(x)).sum()
        x[idx] = orig
        grad[idx] = (plus - minus) / (2.0 * eps)

    return grad


def gradclose(a, b, rtol=1e-4, atol=1e-6):
    """Allclose check that coerces inputs to arrays first."""
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)
