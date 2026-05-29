import numpy as np

def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    """
    Precompute the frequency complex numbers for RoPE.
    Because sometimes you just need to rotate your thoughts.
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(seq_len)
    freqs = np.outer(t, freqs)
    # Convert to complex numbers: e^(i*t*theta)
    freqs_cis = np.exp(1j * freqs)
    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    """
    Apply RoPE to Query or Key tensors.
    x shape: (batch, n_heads, seq_len, head_dim)
    freqs_cis shape: (seq_len, head_dim // 2)
    """
    # Reshape x to complex: (..., head_dim // 2, 2) -> (..., head_dim // 2)
    x_complex = x.reshape(*x.shape[:-1], -1, 2)
    x_complex = x_complex[..., 0] + 1j * x_complex[..., 1]
    
    # Broadcast freqs_cis: (seq_len, dim//2) -> (1, 1, seq_len, dim//2)
    freqs_cis = freqs_cis[np.newaxis, np.newaxis, :, :]
    
    # Rotate
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real
    x_out = np.stack([x_rotated.real, x_rotated.imag], axis=-1)
    return x_out.reshape(*x.shape)
