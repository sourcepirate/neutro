# Core Utility Layers

## Dropout — `neutro/layers/core/dropout.py`

Randomly sets a fraction of inputs to zero during training, preventing co-adaptation:

$$y = \begin{cases} \frac{m \odot x}{1 - p} & \text{training} \\ x & \text{inference} \end{cases}$$

Where $m_i \sim \text{Bernoulli}(1-p)$ is a mask. The scaling by $1/(1-p)$ keeps the expected output magnitude constant.

```python
def forward(self, inputs, training=False):
    if not training:
        return inputs
    self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape)
    return inputs * self.mask / (1 - self.rate)

def backward(self, grad_output):
    return grad_output * self.mask / (1 - self.rate)
```

## Flatten — `neutro/layers/core/flatten.py`

Reshapes a multi-dimensional input into a 2D (batch, features) tensor, preserving the batch dimension:

```python
def forward(self, inputs):
    return inputs.reshape(inputs.shape[0], -1)

def backward(self, grad_output):
    return grad_output.reshape(self.input_shape)
```

## MoE Layer — `neutro/layers/core/moe.py`

### Theory

Mixture-of-Experts (MoE) scales model capacity without proportional compute. A router network selects which "expert" sub-networks to activate for each input token:

$$y = \sum_{i=1}^E g_i(x) \cdot E_i(x)$$

Where $g_i(x)$ is the router's gating weight (typically top-$k$ sparse) and $E_i$ are expert feed-forward networks.

### Router — `neutro/layers/core/moe.py:30`

```python
def forward(self, x):
    logits = np.dot(x, self.params['W'])  # (batch, seq, num_experts)
    weights = softmax(logits, axis=-1)
    # Top-k routing
    top_k_weights, top_k_indices = ...
```

The router learns to assign tokens to the most relevant experts.

## Reparameterization — `neutro/layers/core/reparameterization.py`

Implements the reparameterization trick used in VAEs. A sample from $N(\mu, \sigma^2)$ is:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim N(0, I)$$

This makes the sampling operation differentiable, enabling backpropagation through the stochastic layer.

```python
def forward(self, inputs):
    mu, log_var = inputs
    eps = np.random.randn(*mu.shape)
    return mu + np.exp(0.5 * log_var) * eps
```

## Usage Example

```python
from neutro.layers import Dropout, Flatten, MoELayer

drop = Dropout(rate=0.5)
x = np.random.randn(8, 64)
y = drop(x, training=True)  # 50% of units dropped

flat = Flatten()
x = np.random.randn(8, 4, 4, 16)
y = flat(x)  # (8, 256)

moe = MoELayer(num_experts=8, expert_dim=512, top_k=2)
x = np.random.randn(2, 16, 512)
y = moe(x)
```

## References

- Srivastava, N., et al. (2014). **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**. *JMLR*.
- Shazeer, N., et al. (2017). **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Kingma, D. P., & Welling, M. (2013). **Auto-Encoding Variational Bayes**. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
