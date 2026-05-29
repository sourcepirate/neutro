# Adam Optimizer

## Overview
Adam (Adaptive Moment Estimation) is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.

## Mathematical Formulation
1.  **Momentum**: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2.  **Velocity**: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3.  **Bias Correction**: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
4.  **Update**: $w_{t+1} = w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

## Citations
- Kingma, D. P., & Ba, J. (2014). **Adam: A Method for Stochastic Optimization**. *arXiv preprint arXiv:1412.6980*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
