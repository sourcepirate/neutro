# Grouped-Query Attention (GQA)

## Overview
Grouped-Query Attention generalizes MQA by using an intermediate number of key-value heads (more than one, but fewer than the number of query heads). It provides a balance between the speed of MQA and the quality of MHA.

## Mathematical Formulation
The query heads are divided into $G$ groups. Each group shares a single key and value head.
If there are $H$ query heads and $G$ groups, each group has $H/G$ query heads sharing one KV head.

## Implementation Details
We reshape the query heads into groups and broadcast the corresponding KV heads within each group. This implementation is optimized for modern LLM architectures like Llama 3.

## Citations
- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
