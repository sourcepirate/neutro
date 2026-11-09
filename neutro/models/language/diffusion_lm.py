import numpy as np
from ..base_model import Model
from ...layers.embedding.embedding import Embedding
from ...layers.normalization.rmsnorm import RMSNorm
from ...layers.core.dense import Dense
from ...layers.diffusion.time_mlp import TimeMLPEmbedding
from ...layers.diffusion.dit_block import DiTBlock
from ...layers.diffusion.discrete_scheduler import DiscreteDiffusionScheduler


class MaskedDiffusionLM(Model):
    """Masked diffusion LM (LLaDA/MDLM-style) with DiT backbone.

    Forward corrupts inputs via absorbing [MASK] scheduler when training.
    Sampling iteratively unmasks low-confidence tokens.
    """

    def __init__(self, vocab_size, seq_len, dim=128, n_layers=2, n_heads=4,
                 timesteps=128, mask_id=None, schedule='linear',
                 time_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.mask_id = vocab_size - 1 if mask_id is None else mask_id
        self.scheduler = DiscreteDiffusionScheduler(
            timesteps=timesteps, schedule=schedule, mask_id=self.mask_id)
        self.token_emb = Embedding(vocab_size, dim)
        self.time_mlp = TimeMLPEmbedding(time_dim)
        # Project time cond to model dim for AdaLN.
        self.time_proj = Dense(dim)
        self.blocks = [DiTBlock(dim, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm()
        self.head = Dense(vocab_size)
        self.layers = [self.token_emb, self.time_mlp, self.time_proj,
                       *self.blocks, self.norm, self.head]
        self._last_mask = None

    def build(self, input_shape):
        self.token_emb.build(input_shape)
        self.time_mlp.build((input_shape[0],))
        self.time_proj.build((input_shape[0], self.time_mlp.dim))
        h_shape = (input_shape[0], input_shape[1], self.dim)
        c_shape = (input_shape[0], self.dim)
        for b in self.blocks:
            b.build([h_shape, c_shape])
        self.norm.build(h_shape)
        self.head.build(h_shape)
        super().build(input_shape)

    def forward(self, inputs, training=False):
        if training:
            x0 = np.asarray(inputs)
            t = np.random.randint(0, self.scheduler.timesteps, (x0.shape[0],))
            xt, mask = self.scheduler.q_sample(x0, t)
            self._last_mask = mask
            self._last_t = t
        else:
            xt = np.asarray(inputs)
            t = np.zeros((xt.shape[0],), dtype=np.int64)
        h = self.token_emb(xt, training)
        cond = self.time_proj(self.time_mlp(t, training), training)
        for b in self.blocks:
            h = b([h, cond], training)
        return self.head(self.norm(h, training), training)

    def backward(self, grad):
        grad = self.head.backward(grad)
        grad = self.norm.backward(grad)
        # grad through blocks (each takes [h, cond]); cond path accumulates.
        cond_grads = []
        for b in reversed(self.blocks):
            out = b.backward(grad)
            # DiTBlock.backward returns list [dh, dcond] via tape path.
            if isinstance(out, list):
                grad, dcond = out
                cond_grads.append(dcond)
            else:
                grad = out
        grad = self.token_emb.backward(grad)
        return grad

    def sample(self, prefix=None, seq_len=None, steps=32, temperature=1.0,
               rng=None):
        rng = rng or np.random.default_rng(0)
        seq_len = seq_len or self.seq_len
        batch = 1 if prefix is None else prefix.shape[0]
        if prefix is None:
            x = np.full((batch, seq_len), self.mask_id, dtype=np.int64)
        else:
            x = np.full((batch, seq_len), self.mask_id, dtype=np.int64)
            L = min(prefix.shape[1], seq_len)
            x[:, :L] = np.asarray(prefix)[:, :L]
            fixed = np.zeros_like(x, dtype=bool)
            fixed[:, :L] = True
        else_fixed = fixed if prefix is not None else np.zeros_like(x, dtype=bool)
        # Linear unmasking schedule.
        to_unmask = int(np.sum(~else_fixed))
        per_step = max(1, to_unmask // max(steps, 1))
        for _ in range(steps):
            logits = np.asarray(self.forward(x, training=False))
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            e = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(e) / np.exp(e).sum(axis=-1, keepdims=True)
            pred = np.stack([[rng.choice(self.vocab_size, p=(p / p.sum()))
                              for p in probs[b]]
                             for b in range(batch)], axis=0)
            conf = probs.max(axis=-1)
            conf[else_fixed] = 2.0
            conf[x != self.mask_id] = 2.0
            order = np.argsort(conf, axis=1)
            for b in range(batch):
                n = 0
                for idx in order[b]:
                    if x[b, idx] == self.mask_id and not else_fixed[b, idx]:
                        x[b, idx] = pred[b, idx]
                        n += 1
                        if n >= per_step:
                            break
            if np.all(x != self.mask_id):
                break
        return x
