# Multi-Head Attention (MHA)

## Overview
Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.

## Mathematical Formulation
Given a query $Q$, key $K$, and value $V$, the scaled dot-product attention is:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In MHA, we project $Q, K, V$ into $h$ heads:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

## Implementation Details
In `neutro`, we use NumPy broadcasting to compute all heads in parallel. The input shape is typically `(batch, seq_len, embed_dim)`. We reshape this to `(batch, heads, seq_len, head_dim)` to perform the batched dot product.

## Citations
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. *Advances in Neural Information Processing Systems (NeurIPS)*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
