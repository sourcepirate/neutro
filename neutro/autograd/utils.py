import numpy as np


def broadcast_backward(grad, original_shape):
    """
    Reduce grad to original_shape by summing over broadcasted dimensions.
    Handles leading broadcast dimensions and dimensions where original size is 1.
    """
    grad = np.asarray(grad)
    if grad.shape == original_shape:
        return grad

    original_shape = tuple(original_shape) if not isinstance(original_shape, tuple) else original_shape

    # Handle scalar original shape
    if len(original_shape) == 0:
        return grad.sum().reshape(original_shape)

    # Sum over leading dimensions added by broadcasting
    ndim_diff = grad.ndim - len(original_shape)
    if ndim_diff > 0:
        # Sum over leading axes
        grad = grad.sum(axis=tuple(range(ndim_diff)))
        # After this, grad.ndim == len(original_shape) or could still be larger if original had size 1 dims?

    # Now grad.ndim == len(original_shape) (after leading reduction)
    # For any dimension where original_shape[d]==1 and grad.shape[d]>1, sum over that axis
    # Use keepdims to preserve dimensionality for subsequent reductions
    # Collect axes to sum in one go for efficiency when possible
    # However sequential keepdims sums are clearer and keep shape aligned.
    for i, (gs, os) in enumerate(zip(grad.shape, original_shape)):
        if os == 1 and gs != 1:
            grad = grad.sum(axis=i, keepdims=True)
        elif gs != os and os != 1:
            # Shape mismatch that is not due to broadcasting -> likely error, but try to handle via reshape
            # This can happen if grad was reduced earlier and shapes still mismatch
            # Fall back to explicit reshape after summing all broadcast dims
            pass

    # Final reshape to ensure exact original_shape (handles cases where keepdims produced correct shape)
    if grad.shape != original_shape:
        # If grad is scalar after reductions, broadcast to original_shape with keeping?
        # For safety, try to reshape via broadcasting sum
        try:
            grad = grad.reshape(original_shape)
        except ValueError:
            # As fallback, use numpy broadcast_to logic via sum
            # Sum to original_shape using iterative approach
            # If still mismatch, raise informative error
            raise ValueError(f"broadcast_backward shape mismatch: grad shape {grad.shape} cannot be reduced to {original_shape}")
    return grad


def numeric_grad(f, x, eps=1e-6):
    """
    Compute numeric gradient via central difference.
    f: callable taking numpy array -> scalar (0-d array or float)
    x: numpy array
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    # Use iterator over indices; for large arrays this is O(N) but fine for testing
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        # Ensure scalar outputs
        f_plus = np.asarray(f_plus).sum()
        f_minus = np.asarray(f_minus).sum()
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def gradclose(a, b, rtol=1e-4, atol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)
