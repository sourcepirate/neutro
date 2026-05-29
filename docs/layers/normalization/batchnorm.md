# Batch Normalization

## Overview
Batch Normalization (BatchNorm) accelerates deep network training by reducing internal covariate shift. It normalizes the activations of each layer for each mini-batch.

## Mathematical Formulation
For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$:
1.  **Mean**: $\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i$
2.  **Variance**: $\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$
3.  **Normalize**: $\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$
4.  **Scale and Shift**: $y_i = \gamma \hat{x}_i + \beta$

## Implementation Details
`neutro` tracks running means and variances during training to use for inference. The backward pass involves calculating gradients for $\gamma$, $\beta$, and the input $x$ with respect to the batch statistics.

## Citations
- Ioffe, S., & Szegedy, C. (2015). **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
