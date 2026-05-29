# Multi-Query Attention (MQA)

## Overview
Multi-Query Attention is a variation of multi-head attention where the keys and values are shared across all query heads. This significantly reduces memory bandwidth during incremental decoding.

## Mathematical Formulation
While $Q$ has $h$ heads, $K$ and $V$ have only 1 head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW^K, VW^V)$$
The key and value projections are shared across all $i \in \{1, \dots, h\}$.

## Implementation Details
We broadcast the single $K$ and $V$ heads across all $Q$ heads during the attention score calculation. This reduces the KV cache size by a factor of $h$ in inference.

## Citations
- Shazeer, N. (2019). **Fast Transformer Decoding: One Write-Head is All You Need**. *arXiv preprint arXiv:1911.02150*. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
