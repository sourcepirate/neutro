# FlashAttention

## Overview
FlashAttention is a fast and memory-efficient algorithm for computing exact attention. It reduces the memory complexity from $O(N^2)$ to $O(N)$ by avoiding the explicit calculation of the full $N \times N$ attention matrix. Instead, it uses **tiling** and a modified **online softmax** algorithm.

## Mathematical Formulation
The standard attention computes $O = \text{softmax}(QK^T)V$. FlashAttention computes this in tiles.

### Online Softmax
To compute softmax over tiles, we maintain running statistics for each row $i$:
1.  **Running Max ($M_i$):** The maximum attention score seen so far.
2.  **Running Sum ($L_i$):** The sum of exponentials relative to $M_i$.

When a new tile $j$ is processed:
-   Compute local max $m_{ij}$ and local sum $l_{ij}$.
-   Update global max: $M_{i}^{new} = \max(M_i, m_{ij})$.
-   Update global sum: $L_{i}^{new} = e^{M_i - M_i^{new}} L_i + e^{m_{ij} - M_i^{new}} l_{ij}$.
-   Rescale partial output: $O_i = e^{M_i - M_i^{new}} O_i + e^{m_{ij} - M_i^{new}} (P_{ij} V_j)$.

## Implementation Details
In `neutro`, the `FlashAttention` layer implements the tiled forward pass. Even though NumPy runs on CPU, this implementation demonstrates the principle of constant memory overhead for the attention scores. The block sizes `block_size_r` and `block_size_c` can be tuned to simulate memory constraints.

## Citations
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. *Advances in Neural Information Processing Systems (NeurIPS)*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, T. (2023). **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**. *arXiv preprint arXiv:2307.08691*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
