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
        """
        Updates the cache with new K and V, and returns the full history.
        k, v: (batch, num_heads, 1, head_dim) - usually just one token during generation
        """
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = k
            self.v_cache[layer_id] = v
        else:
            # Concatenate along the sequence dimension (axis 2)
            self.k_cache[layer_id] = np.concatenate([self.k_cache[layer_id], k], axis=2)
            self.v_cache[layer_id] = np.concatenate([self.v_cache[layer_id], v], axis=2)
            
        return self.k_cache[layer_id], self.v_cache[layer_id]

    def reset(self):
        self.k_cache = {}
        self.v_cache = {}
