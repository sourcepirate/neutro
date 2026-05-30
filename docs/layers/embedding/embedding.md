# Embedding Layers

## Theory

### Token Embedding — `neutro/layers/embedding/embedding.py`

An embedding layer maps discrete tokens (integers) to dense vectors:

$$x_i = W[\text{token}_i]$$

Where $W \in \mathbb{R}^{V \times D}$ is a learnable matrix, $V$ is the vocabulary size, and $D$ is the embedding dimension. The forward pass is a simple lookup:

```python
def forward(self, inputs):
    return self.params['W'][inputs]  # (batch, seq_len, embed_dim)
```

The backward pass uses `np.add.at` to accumulate gradients back to the embedding matrix:

```python
def backward(self, grad_output):
    self.grads['W'] = np.zeros_like(self.params['W'])
    np.add.at(self.grads['W'], self.inputs, grad_output)
    return grad_output
```

### TimeEmbedding — `neutro/layers/embedding/time_embedding.py`

Projects scalar timesteps (e.g., diffusion timesteps) into a high-dimensional space using sinusoidal encoding followed by a learnable MLP projection.

## Usage Example

```python
from neutro.layers import Embedding

emb = Embedding(vocab_size=10000, embed_dim=512)
tokens = np.array([[1, 5, 23, 42]])  # (batch, seq_len)
x = emb(tokens)  # (1, 4, 512)
```

## References

- Mikolov, T., et al. (2013). **Efficient Estimation of Word Representations in Vector Space**. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
