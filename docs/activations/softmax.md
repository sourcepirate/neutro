# Softmax Activation

## Overview
The Softmax activation function transforms a vector of real numbers into a probability distribution. Unlike element-wise activations like ReLU, the gradient of Softmax depends on all components of the input vector.

## Mathematical Formulation
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

## Gradient and Jacobian
The derivative of the $i$-th output with respect to the $j$-th input is:
$$\frac{\partial y_i}{\partial x_j} = y_i (\delta_{ij} - y_j)$$
where $\delta_{ij}$ is the Kronecker delta.

## Optimization: `gradient_fast`
In `neutro`, we implement a specialized `gradient_fast` method in `neutro/activations/softmax.py`. This method efficiently computes the product of the Jacobian matrix and the upstream gradient (Jacobian-Vector Product) without explicitly constructing the full $N \times N$ Jacobian matrix, saving memory and computation:
$$\text{grad\_input} = y * (\text{grad\_output} - \sum(y * \text{grad\_output}))$$

## References
- Deep Learning (Goodfellow, Bengio, Courville). **Chapter 6: Deep Feedforward Networks**. [Deep Learning Book](https://www.deeplearningbook.org/contents/mlp.html)
- CS231n: **Softmax Classifier**. [Stanford Notes](https://cs231n.github.io/linear-classify/#softmax)
