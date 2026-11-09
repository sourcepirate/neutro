from ..base_model import Model
from ...layers.convolutional.conv2d import Conv2D
from ...layers.core.merging import Concatenate
from ...layers.diffusion.time_mlp import TimeMLPEmbedding
from ...layers.core.dense import Dense
from ...layers.pooling.maxpooling2d import MaxPooling2D
from ...layers.pooling.upsampling2d import UpSampling2D


class UNet(Model):
    """Simplified UNet for diffusion with time conditioning."""

    def __init__(self, input_channels, base_filters=64, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.base_filters = base_filters
        self.time_mlp = TimeMLPEmbedding(time_dim)
        self.time_proj = Dense(base_filters * 4)
        
        # Encoder
        self.enc1 = Conv2D(base_filters, 3, padding='same', activation='relu')
        self.pool1 = MaxPooling2D((2, 2))
        self.enc2 = Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        
        # Bottleneck
        self.bottleneck = Conv2D(base_filters * 4, 3, padding='same', activation='relu')
        
        # Decoder
        self.up1 = UpSampling2D((2, 2))
        self.concat1 = Concatenate(axis=-1)
        self.dec1 = Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        
        self.up2 = UpSampling2D((2, 2))
        self.concat2 = Concatenate(axis=-1)
        self.dec2 = Conv2D(base_filters, 3, padding='same', activation='relu')
        
        # Output
        self.final_conv = Conv2D(input_channels, 1, padding='same')
        
        self.layers = [
            self.enc1, self.pool1, self.enc2, self.pool2,
            self.time_mlp, self.time_proj,
            self.bottleneck,
            self.up1, self.concat1, self.dec1,
            self.up2, self.concat2, self.dec2,
            self.final_conv
        ]

    def build(self, input_shape):
        self.input_shape = input_shape
        self.built = True

    def forward(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: ``[x, t]`` where ``x`` is image and ``t`` is timestep.
        """
        x, t = inputs

        # Encoder
        h1 = self.enc1(x, training)
        p1 = self.pool1(h1)
        h2 = self.enc2(p1, training)
        p2 = self.pool2(h2)

        # Time conditioning: project timestep emb to bottleneck channels,
        # added to bottleneck output (B,H,W,C).
        t_emb = self.time_mlp(t, training)          # (B, time_dim) Tensor
        t_bias = self.time_proj(t_emb, training)    # (B, C) Tensor
        # Broadcast over spatial dims: (B,1,1,C) + (B,H,W,C).
        tb = t_bias.reshape((t_bias.shape[0], 1, 1, t_bias.shape[1]))

        # Bottleneck
        b = self.bottleneck(p2, training)
        b = b + tb
        
        # Decoder
        u1 = self.up1(b)
        c1 = self.concat1([u1, h2])
        d1 = self.dec1(c1, training)
        
        u2 = self.up2(d1)
        c2 = self.concat2([u2, h1])
        d2 = self.dec2(c2, training)
        
        return self.final_conv(d2, training)

    def backward(self, grad_output):
        # This needs manual backprop since it's not sequential
        g_final = self.final_conv.backward(grad_output)
        
        g_d2 = self.dec2.backward(g_final)
        g_u2, g_h1_dec = self.concat2.backward(g_d2)
        g_d1_up = self.up2.backward(g_u2)
        
        g_d1 = self.dec1.backward(g_d1_up)
        g_u1, g_h2_dec = self.concat1.backward(g_d1)
        g_b_up = self.up1.backward(g_u1)

        # Split time-conditioning add: b_total = bottleneck(p2) + broadcast(t_bias).
        # g_b_up is grad w.r.t. b_total (B,H,W,Cbottleneck).
        import numpy as _np
        g_b_total = _np.asarray(g_b_up)
        g_t_bc = g_b_total.sum(axis=(1, 2))  # (B, Cbottleneck)
        g_t_emb = self.time_proj.backward(g_t_bc)
        # time_mlp input is integer timesteps; grad not needed downstream.
        try:
            self.time_mlp.backward(_np.asarray(g_t_emb))
        except Exception:
            pass

        g_b = self.bottleneck.backward(g_b_total)
        g_p2 = self.pool2.backward(g_b)
        # h2 was used in pool2 AND concat1
        g_h2_total = g_p2 + g_h2_dec
        g_p1 = self.enc2.backward(g_h2_total)
        
        g_h1_from_pool = self.pool1.backward(g_p1)
        # h1 was used in pool1 AND concat2
        g_h1_total = g_h1_from_pool + g_h1_dec
        
        return self.enc1.backward(g_h1_total)
