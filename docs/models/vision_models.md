# Vision Models

## UNet — `neutro/models/vision/unet.py`

A simplified UNet for diffusion-based image generation. The architecture is:

**Encoder**: `Conv2D → MaxPool → Conv2D → MaxPool → Conv2D`
**Decoder**: `UpSample → Concat (skip) → Conv2D → UpSample → Concat (skip) → Conv2D`

The UNet is a **subclassed model** with a custom forward that takes `[x, t]` (image + timestep):

```python
def forward(self, inputs, training=False):
    x, t = inputs
    h1 = self.enc1(x, training)           # (B, H, W, base_filters)
    p1 = self.pool1(h1)                   # (B, H/2, W/2, base_filters)
    h2 = self.enc2(p1, training)          # (B, H/2, W/2, base_filters*2)
    p2 = self.pool2(h2)                   # (B, H/4, W/4, base_filters*2)
    b = self.bottleneck(p2, training)     # (B, H/4, W/4, base_filters*4)

    u1 = self.up1(b)                      # (B, H/2, W/2, base_filters*4)
    c1 = self.concat1([u1, h2])           # Skip connection
    d1 = self.dec1(c1, training)          # (B, H/2, W/2, base_filters*2)

    u2 = self.up2(d1)                     # (B, H, W, base_filters*2)
    c2 = self.concat2([u2, h1])           # Skip connection
    d2 = self.dec2(c2, training)          # (B, H, W, base_filters)
    return self.final_conv(d2, training)  # (B, H, W, input_channels)
```

The backward pass manually routes gradients through the skip connections and concatenation splits.

## VAE — `neutro/models/vision/vae.py`

Variational Autoencoder with:

- **Encoder**: Conv2D → Dense → outputs mu, log_var (latent distribution parameters).
- **Decoder**: Dense → Conv2D Transpose (via UpSampling2D + Conv2D) → reconstructs input.
- **Loss**: `VAELoss` combines reconstruction loss + KL divergence.

```python
def forward(self, x):
    mu, log_var = self.encoder(x)
    z = self.reparameterize(mu, log_var)
    x_recon = self.decoder(z)
    return x_recon, mu, log_var
```

## DiffusionModel — `neutro/models/vision/diffusion_model.py`

Wraps a UNet with the diffusion process:

- **Training**: Adds noise to image, asks UNet to predict noise via `q_sample`.
- **Inference**: Iteratively denoises from pure noise via `p_sample`.
- **Loss**: MSE between predicted noise and actual noise.

```python
def forward(self, x_start, training=False):
    if training:
        t = np.random.randint(0, timesteps, (batch,))
        noise = np.random.normal(size=x_start.shape)
        x_t = self.diffusion.q_sample(x_start, t, noise)
        predicted_noise = self.unet([x_t, t])
        return predicted_noise
```

## AlexNet — `neutro/models/vision/alexnet.py`

A simplified AlexNet: Conv → Pool → Conv → Pool → Conv → Dense → Dense.

## VGG — `neutro/models/vision/vgg.py`

VGG-style architecture: blocks of Conv2D(3×3) with increasing filters, interleaved with MaxPooling, ending with Dense layers.

## Usage Example

```python
from neutro.models.vision import UNet, VAE, DiffusionModel
import numpy as np

# UNet
unet = UNet(input_channels=1, base_filters=32)
x, t = np.random.randn(2, 16, 16, 1), np.array([5, 3])
out = unet.forward([x, t])

# Diffusion
model = DiffusionModel(unet, timesteps=1000)
sample = model.sample(shape=(1, 16, 16, 1))
```

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation**. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Kingma, D. P., & Welling, M. (2013). **Auto-Encoding Variational Bayes**. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
