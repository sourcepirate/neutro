import numpy as np
from .base import Loss

class SparseCategoricalCrossentropy(Loss):
    """
    Sparse Categorical Crossentropy loss.
    y_true should be integer labels.
    y_pred should be probabilities (after softmax).
    """
    def __call__(self, y_true, y_pred):
        # y_true: (batch, seq) or (batch,)
        # y_pred: (batch, seq, vocab) or (batch, vocab)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Flatten if necessary
        if y_true.ndim < y_pred.ndim - 1:
             # handle cases where y_true is missing the seq dimension if y_pred has it
             pass 

        # We use advanced indexing to pick the probabilities of the true labels
        # Reshape to 2D for easier indexing: (N, C)
        n_samples = np.prod(y_true.shape)
        vocab_size = y_pred.shape[-1]
        
        y_pred_flat = y_pred.reshape(-1, vocab_size)
        y_true_flat = y_true.reshape(-1).astype(int)
        
        prob_of_true = y_pred_flat[np.arange(n_samples), y_true_flat]
        return -np.mean(np.log(prob_of_true))

    def gradient(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n_samples = np.prod(y_true.shape)
        vocab_size = y_pred.shape[-1]
        
        y_pred_flat = y_pred.reshape(-1, vocab_size)
        y_true_flat = y_true.reshape(-1).astype(int)
        
        grad = np.zeros_like(y_pred_flat)
        # dL/dp = -1/p for the true label
        grad[np.arange(n_samples), y_true_flat] = -1.0 / y_pred_flat[np.arange(n_samples), y_true_flat]
        
        return (grad / n_samples).reshape(y_pred.shape)
