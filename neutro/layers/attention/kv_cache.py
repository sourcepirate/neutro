import numpy as np

class KVCache:
    """
    A simple KV Cache for educational purposes.
    It stores the Keys and Values of previous tokens to avoid recomputing them
    during autoregressive generation.
    
    In this naive version, we grow the cache dynamically.
    """
    def __init__(self):
        self.k_cache = {} # layer_id -> k_tensor
        self.v_cache = {} # layer_id -> v_tensor

    def update(self, k, v, layer_id):
        """Update the cache with new K/V and return the full history.

        Args:
            k, v: tensors of shape ``(B, H, new_tokens, d)``. During
                generation ``new_tokens`` is usually 1.
            layer_id: identifies which attention layer owns this cache.

        Returns:
            ``(k_full, v_full)`` with all cached tokens along axis 2.

        The concatenation goes through ``autograd_ops.concatenate`` when the
        inputs are ``Tensor`` objects, so the tape is not silently broken by a
        NumPy conversion. During ordinary inference the cache holds plain
        arrays or tensors with no active tape.
        """
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = k
            self.v_cache[layer_id] = v
        else:
            self.k_cache[layer_id] = self._concat(self.k_cache[layer_id], k)
            self.v_cache[layer_id] = self._concat(self.v_cache[layer_id], v)

        return self.k_cache[layer_id], self.v_cache[layer_id]

    @staticmethod
    def _concat(previous, new):
        """Concatenate along the sequence axis without dropping autograd."""
        from neutro.autograd import Tensor, ops as autograd_ops

        if isinstance(previous, Tensor) or isinstance(new, Tensor):
            return autograd_ops.concatenate([previous, new], axis=2)
        return np.concatenate([previous, new], axis=2)

    def reset(self):
        self.k_cache = {}
        self.v_cache = {}
