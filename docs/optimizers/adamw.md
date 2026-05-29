# AdamW Optimizer

## Overview
AdamW is a variation of the Adam optimizer that decouples weight decay from the gradient update. This leads to better generalization and is the standard optimizer for training modern Transformers.

## Mathematical Formulation
In standard Adam with L2 regularization, the weight decay is added to the gradient. In AdamW, the weight decay is applied directly to the weights:
1.  **Gradient**: $g_t = \nabla f(w_t)$
2.  **Momentum**: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3.  **Velocity**: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4.  **Update**: $w_{t+1} = w_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_t \right)$
where $\lambda$ is the weight decay coefficient.

## Implementation Details
`neutro` implements AdamW in `neutro/optimizers/adamw.py`. It correctly handles the decoupled weight decay step separately from the adaptive learning rate update.

## Citations
- Loshchilov, I., & Hutter, F. (2017). **Decoupled Weight Decay Regularization**. *International Conference on Learning Representations (ICLR)*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
