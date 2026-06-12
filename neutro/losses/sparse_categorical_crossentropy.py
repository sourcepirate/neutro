import numpy as np
from .base import Loss

class SparseCategoricalCrossentropy(Loss):
    """
    Sparse Categorical Crossentropy loss.
    
    y_true should be integer labels.
    
    When from_logits=False (default): y_pred should be probabilities (after softmax).
    When from_logits=True: y_pred should be raw logits; softmax is applied internally
    and the correct combined CE+softmax gradient is computed.
    """
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def _to_probs(self, y_pred):
        if self.from_logits:
            y_pred_exp = np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True))
            return y_pred_exp / np.sum(y_pred_exp, axis=-1, keepdims=True)
        return y_pred

    def __call__(self, y_true, y_pred):
        # y_true: (batch, seq) or (batch,)
        # y_pred: (batch, seq, vocab) or (batch, vocab)
        y_pred = self._to_probs(y_pred)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Flatten if necessary
        if y_true.ndim < y_pred.ndim - 1:
            raise ValueError(
                f"Shape mismatch: y_true has {y_true.ndim} dimension(s) but y_pred has "
                f"{y_pred.ndim} dimensions. y_true must have exactly y_pred.ndim - 1 dimensions."
            )

        # We use advanced indexing to pick the probabilities of the true labels
        # Reshape to 2D for easier indexing: (N, C)
        n_samples = np.prod(y_true.shape)
        vocab_size = y_pred.shape[-1]
        
        y_pred_flat = y_pred.reshape(-1, vocab_size)
        y_true_flat = y_true.reshape(-1).astype(int)
        
        prob_of_true = y_pred_flat[np.arange(n_samples), y_true_flat]
        return -np.mean(np.log(prob_of_true))

    def gradient(self, y_true, y_pred):
        n_samples = np.prod(y_true.shape)
        vocab_size = y_pred.shape[-1]
        y_true_flat = y_true.reshape(-1).astype(int)

        if self.from_logits:
            # Combined softmax + cross-entropy gradient: dL/dz_i = p_i - 1_{i==t}
            y_pred_exp = np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True))
            softmax = y_pred_exp / np.sum(y_pred_exp, axis=-1, keepdims=True)
            softmax_flat = softmax.reshape(-1, vocab_size)
            grad = softmax_flat.copy()
            grad[np.arange(n_samples), y_true_flat] -= 1
            return (grad / n_samples).reshape(y_pred.shape)
        else:
            # Gradient of -log(p) w.r.t. probabilities: dL/dp_i = -1/p_t for true label, 0 elsewhere
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            y_pred_flat = y_pred.reshape(-1, vocab_size)
            grad = np.zeros_like(y_pred_flat)
            grad[np.arange(n_samples), y_true_flat] = -1.0 / y_pred_flat[np.arange(n_samples), y_true_flat]
            return (grad / n_samples).reshape(y_pred.shape)
