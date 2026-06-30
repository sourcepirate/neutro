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
    Apply RoPE to Query or Key tensors using real-valued rotation.
    x shape: (batch, n_heads, seq_len, head_dim)
    freqs_cis shape: (seq_len, head_dim // 2)
    """
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]

    cos = freqs_cis.real
    sin = freqs_cis.imag
    cos = cos[np.newaxis, np.newaxis, :, :]
    sin = sin[np.newaxis, np.newaxis, :, :]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    try:
        from neutro.autograd.ops import concatenate
        out = concatenate([out_even[..., None], out_odd[..., None]], axis=-1)
    except ImportError:
        out = np.stack([np.asarray(out_even), np.asarray(out_odd)], axis=-1)
    return out.reshape(*x.shape)
