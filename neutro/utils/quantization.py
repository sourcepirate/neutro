"""NF4 and uniform quantization utilities for QLoRA.

References:
  - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023.
  - NF4 codebook is optimal for N(0,1) weights (quantile quantization).

All operations are pure NumPy.
"""
import numpy as np

# ---------------------------------------------------------------------------
# NF4 codebook — 16 levels optimized for normally-distributed weights.
# Values are the 4-bit NormalFloat quantiles from the QLoRA paper.
# ---------------------------------------------------------------------------
NF4_CODEBOOK = np.array([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=np.float64)

NF4_CODEBOOK_SIZE = 16


# ---------------------------------------------------------------------------
# NF4 block-wise quantize / dequantize
# ---------------------------------------------------------------------------

def quantize_nf4(W, block_size=64):
    """Quantize weight matrix ``W`` to 4-bit NormalFloat (NF4).

    Block-wise absmax normalization is applied: each contiguous block of
    ``block_size`` elements is scaled by its own absmax before quantizing.

    Args:
        W: ndarray of any shape, float weights.
        block_size: number of elements per quantization block.

    Returns:
        codes: ndarray of same shape as W, dtype uint8, values in [0, 15]
               (index into NF4_CODEBOOK).
        absmax: ndarray of shape (num_blocks,), per-block absmax scale.
        orig_shape: tuple, original shape (needed for dequantize).
    """
    W_flat = W.ravel().astype(np.float64)
    n = W_flat.size
    num_blocks = int(np.ceil(n / block_size))

    # Pad to multiple of block_size so reshape is clean.
    pad_len = num_blocks * block_size - n
    if pad_len > 0:
        W_padded = np.concatenate([W_flat, np.zeros(pad_len, dtype=np.float64)])
    else:
        W_padded = W_flat

    W_blocks = W_padded.reshape(num_blocks, block_size)  # (B, block_size)

    # Per-block absmax scale.
    absmax = np.max(np.abs(W_blocks), axis=1)  # (B,)
    # Avoid division by zero for all-zero blocks.
    absmax = np.maximum(absmax, 1e-8)

    # Normalize each block to [-1, 1].
    W_norm = W_blocks / absmax[:, None]  # (B, block_size)
    W_norm = np.clip(W_norm, -1.0, 1.0)

    # Map each normalized value to nearest NF4 codebook entry.
    # Compute distance to each codebook value: (B, block_size, 16) -> argmin.
    # For efficiency with small codebook, use broadcasting.
    # W_norm shape (B, block_size) -> expand last dim
    diff = np.abs(W_norm[:, :, None] - NF4_CODEBOOK[None, None, :])  # (B, bs, 16)
    codes_blocks = np.argmin(diff, axis=2).astype(np.uint8)  # (B, block_size)

    codes_flat = codes_blocks.ravel()[:n]
    codes = codes_flat.reshape(W.shape)

    return codes, absmax, W.shape


def dequantize_nf4(codes, absmax, block_size=64, shape=None):
    """Dequantize NF4 codes back to float.

    Args:
        codes: ndarray, uint8 values in [0, 15] (NF4 indices).
        absmax: ndarray, per-block absmax from quantize_nf4.
        block_size: must match the value used in quantize_nf4.
        shape: original shape tuple (if codes was reshaped).

    Returns:
        W_deq: ndarray float, same shape as original W.
    """
    if shape is None:
        shape = codes.shape
    n = int(np.prod(shape))
    num_blocks = absmax.shape[0]

    pad_len = num_blocks * block_size - n
    codes_flat = codes.ravel().astype(np.int64)
    if pad_len > 0:
        codes_flat = np.concatenate([codes_flat, np.zeros(pad_len, dtype=np.int64)])

    codes_blocks = codes_flat.reshape(num_blocks, block_size)  # (B, bs)

    # Look up codebook values.
    W_norm_blocks = NF4_CODEBOOK[codes_blocks]  # (B, bs)

    # De-normalize.
    W_blocks = W_norm_blocks * absmax[:, None]  # (B, bs)

    W_flat = W_blocks.ravel()[:n]
    return W_flat.reshape(shape)


# ---------------------------------------------------------------------------
# Double quantization (quantize the absmax scales themselves to 8-bit)
# ---------------------------------------------------------------------------

def quantize_absmax(absmax, block_size=256):
    """Double-quantize absmax scales to 8-bit uniform.

    Mirrors the QLoRA double-quantization: the block absmax values are
    themselves quantized to int8 per ``block_size`` super-block.

    Args:
        absmax: ndarray (num_blocks,) float scales.
        block_size: super-block size for the second-level quant.

    Returns:
        q_absmax: ndarray uint8
        dq_absmax: ndarray float (dequantized absmax, for convenience)
        scales2: ndarray float per super-block scale
        absmax_shape: tuple
    """
    flat = absmax.ravel().astype(np.float64)
    n = flat.size
    num_blocks = int(np.ceil(n / block_size))
    pad_len = num_blocks * block_size - n
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=flat.dtype)])
    blocks = flat.reshape(num_blocks, block_size)  # (S, sb)

    scales2 = np.max(np.abs(blocks), axis=1)  # (S,)
    scales2 = np.maximum(scales2, 1e-8)

    # 8-bit uniform quant: map [-scales2, scales2] or [0, scales2] ?
    # absmax values are positive, so quantize in [0, scales2].
    normed = blocks / scales2[:, None]  # [0, 1]
    normed = np.clip(normed, 0, 1)
    q_blocks = np.round(normed * 255).astype(np.uint8)  # (S, sb)

    q_absmax = q_blocks.ravel()[:n].reshape(absmax.shape)
    # Dequantize for convenience
    dq = dequantize_absmax(q_absmax, scales2, block_size, absmax.shape)
    return q_absmax, dq, scales2, absmax.shape


def dequantize_absmax(q_absmax, scales2, block_size=256, shape=None):
    """Dequantize double-quantized absmax."""
    if shape is None:
        shape = q_absmax.shape
    n = int(np.prod(shape))
    flat = q_absmax.ravel().astype(np.float64)
    num_blocks = scales2.shape[0]
    pad_len = num_blocks * block_size - n
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len)])
    blocks = flat.reshape(num_blocks, block_size)
    # De-norm
    dq_blocks = blocks / 255.0 * scales2[:, None]
    dq_flat = dq_blocks.ravel()[:n]
    return dq_flat.reshape(shape)


# ---------------------------------------------------------------------------
# Simple uniform int8 / int4 quantization (helper, not NF4)
# ---------------------------------------------------------------------------

def quantize_uniform(W, bits=8, block_size=64):
    """Block-wise uniform affine quantization (for comparison / ablations)."""
    W_flat = W.ravel().astype(np.float64)
    n = W_flat.size
    num_blocks = int(np.ceil(n / block_size))
    pad_len = num_blocks * block_size - n
    if pad_len > 0:
        W_flat = np.concatenate([W_flat, np.zeros(pad_len)])
    blocks = W_flat.reshape(num_blocks, block_size)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.maximum(absmax, 1e-8)
    levels = 2 ** (bits - 1) - 1
    normed = blocks / absmax[:, None]  # [-1, 1]
    q_blocks = np.round(np.clip(normed, -1, 1) * levels).astype(np.int8)
    q_flat = q_blocks.ravel()[:n]
    return q_flat.reshape(W.shape), absmax, W.shape


def dequantize_uniform(codes, absmax, bits=8, block_size=64, shape=None):
    if shape is None:
        shape = codes.shape
    n = int(np.prod(shape))
    num_blocks = absmax.shape[0]
    levels = 2 ** (bits - 1) - 1
    flat = codes.ravel().astype(np.float64)
    pad_len = num_blocks * block_size - n
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len)])
    blocks = flat.reshape(num_blocks, block_size)
    w_blocks = blocks / levels * absmax[:, None]
    w_flat = w_blocks.ravel()[:n]
    return w_flat.reshape(shape)
