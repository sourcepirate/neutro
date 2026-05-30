# Utilities

## conv_utils — `neutro/utils/conv_utils.py`

### im2col and col2im

These functions implement the image-to-column transformation that converts convolution into matrix multiplication:

- **im2col**: Unrolls each filter-sized patch of the input into a column of a matrix. The output matrix has shape `(kernel_size * channels, output_spatial_size)`.
- **col2im**: The inverse operation — redistributes gradients from the column matrix back to the input volume shape.

```python
def im2col(x, kernel_size, strides, padding='valid'):
    # Fancy indexing to extract sliding windows
    ...

def col2im(grad_cols, input_shape, kernel_size, strides):
    # Accumulate gradients back to input positions
    ...
```

Used by `Conv2D` and `MaxPooling2D` for efficient forward/backward computation.

## rope_utils — `neutro/utils/rope_utils.py`

### Rotary Position Embedding (RoPE)

RoPE encodes position information by rotating query and key vectors in attention:

$$\text{RoPE}(x, m) = x \cdot \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

```python
def precompute_freqs_cis(dim, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    t = np.arange(max_seq_len)
    return np.exp(1j * np.outer(t, freqs))
```

- No learned parameters — positions are encoded via rotation.
- Used in Llama, GPT-NeoX, and many modern LLMs.

## diffusion_utils — `neutro/utils/diffusion_utils.py`

Implements the forward diffusion process (adding noise) for DDPM:

```python
class GaussianDiffusion:
    def q_sample(self, x_start, t, noise=None):
        # q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        sqrt_alpha_bar = np.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus = np.sqrt(1 - self.alphas_cumprod[t])
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def p_sample(self, model, x_t, t):
        # Reverse step: denoise x_t using the model
        predicted_noise = model(x_t, t)
        ...
```

## visualization — `neutro/utils/visualization.py`

Provides `plot_attention_weights` for visualizing attention patterns as heatmaps:

```python
def plot_attention_weights(attention_weights, tokens, layer_name=None):
    # Matplotlib heatmap of attention scores
    ...
```

## Usage Example

```python
from neutro.utils.rope_utils import precompute_freqs_cis
from neutro.utils.conv_utils import im2col

freqs = precompute_freqs_cis(dim=64, max_seq_len=512)
cols = im2col(x, kernel_size=(3, 3), strides=1, padding='same')
```

## References

- Su, J., et al. (2021). **RoFormer: Enhanced Transformer with Rotary Position Embedding**. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
