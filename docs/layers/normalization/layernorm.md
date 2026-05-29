# Layer Normalization

## Overview
Layer Normalization (LayerNorm) is a normalization technique that computes the mean and variance for each individual sample across all its features, rather than across a batch. This makes it ideal for recurrent neural networks and Transformers.

## Mathematical Formulation
Unlike BatchNorm, LayerNorm normalizes across the features $H$:
$$\mu = \frac{1}{H} \sum_{i=1}^H x_i$$
$$\sigma = \sqrt{\frac{1}{H} \sum_{i=1}^H (x_i - \mu)^2 + \epsilon}$$
$$\hat{x} = \frac{x - \mu}{\sigma}$$
$$y = \gamma \hat{x} + \beta$$

## Implementation Details
In `neutro`, LayerNorm is used extensively in the `TransformerBlock`. It is independent of the batch size and works the same way during training and inference.

## Citations
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). **Layer Normalization**. *arXiv preprint arXiv:1607.06450*. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
