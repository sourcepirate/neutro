import numpy as np


def broadcast_backward(grad, original_shape):
    if grad.shape == original_shape:
        return grad
    n_leading = grad.ndim - len(original_shape)
    if n_leading > 0:
        grad = grad.sum(axis=tuple(range(n_leading)))
    for i, (gs, os) in enumerate(zip(grad.shape, original_shape)):
        if os == 1 and gs > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(original_shape)


def numeric_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        orig = x[idx].copy()
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def gradclose(a, b, rtol=1e-4, atol=1e-6):
    return np.allclose(a, b, rtol=rtol, atol=atol)
