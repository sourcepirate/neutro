# SGD with Momentum

## Overview
Stochastic Gradient Descent (SGD) is a simple but effective approach to discriminative learning of linear classifiers. Momentum and Nesterov Momentum help accelerate gradients in the right direction.

## Mathematical Formulation
1.  **Velocity**: $v_t = \mu v_{t-1} - \alpha g_t$
2.  **Update**: $w_{t+1} = w_t + v_t$

For **Nesterov Momentum**:
1.  $v_t = \mu v_{t-1} - \alpha g_t$
2.  $w_{t+1} = w_t + \mu v_t - \alpha g_t$

## Citations
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). **On the importance of initialization and momentum in deep learning**. *International Conference on Machine Learning (ICML)*. [PMLR Link](http://proceedings.mlr.press/v28/sutskever13.html)
