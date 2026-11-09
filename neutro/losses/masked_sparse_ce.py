import numpy as np
from .base import Loss


class MaskedSparseCrossentropy(Loss):
    """Cross-entropy averaged only over masked positions.

    y_true: (B, S) int original tokens; y_pred: (B, S, V) logits/probs.
    Mask is read from ``model._last_mask`` set during training forward.
    Falls back to full average when no mask is present.
    """

    def __init__(self, model, from_logits=True):
        self.model = model
        self.from_logits = from_logits

    def _probs(self, y_pred):
        y_pred = np.asarray(y_pred)
        if self.from_logits:
            e = np.exp(y_pred - y_pred.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        return y_pred

    def _mask(self, y_true):
        m = getattr(self.model, '_last_mask', None)
        if m is None:
            return np.ones_like(np.asarray(y_true), dtype=bool)
        return np.asarray(m, dtype=bool)

    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        probs = np.clip(self._probs(y_pred), 1e-15, 1 - 1e-15)
        n = y_true.size
        flat_p = probs.reshape(-1, probs.shape[-1])
        flat_t = y_true.reshape(-1).astype(int)
        nll = -np.log(flat_p[np.arange(n), flat_t]).reshape(y_true.shape)
        mask = self._mask(y_true)
        denom = max(mask.sum(), 1)
        return float((nll * mask).sum() / denom)

    def gradient(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)
        n = y_true.size
        flat_t = y_true.reshape(-1).astype(int)
        V = y_pred_np.shape[-1]
        if self.from_logits:
            e = np.exp(y_pred_np - y_pred_np.max(axis=-1, keepdims=True))
            sm = e / e.sum(axis=-1, keepdims=True)
            grad = sm.reshape(-1, V).copy()
            grad[np.arange(n), flat_t] -= 1
            grad = grad.reshape(y_pred_np.shape)
        else:
            p = np.clip(y_pred_np, 1e-15, 1 - 1e-15)
            flat_p = p.reshape(-1, V)
            grad = np.zeros_like(flat_p)
            grad[np.arange(n), flat_t] = -1.0 / flat_p[np.arange(n), flat_t]
            grad = grad.reshape(y_pred_np.shape)
        mask = self._mask(y_true).astype(float)
        denom = max(mask.sum(), 1)
        # Broadcast mask over vocab dim.
        while mask.ndim < grad.ndim:
            mask = mask[..., None]
        return (grad * mask / denom)
