import numpy as np
from neutro.models.language.qwen import Qwen2Tiny, Qwen3MoETiny
from neutro.models.language.diffusion_lm import MaskedDiffusionLM
from neutro.losses.masked_sparse_ce import MaskedSparseCrossentropy


def test_qwen_moe_smoke():
    m = Qwen3MoETiny(40, 8, dim=16, n_layers=1, n_heads=2,
                     n_experts=4, top_k=2)
    x = np.random.randint(0, 40, (1, 8))
    assert np.asarray(m.predict(x)).shape == (1, 8, 40)


def test_diffusion_lm_train_and_sample():
    m = MaskedDiffusionLM(vocab_size=12, seq_len=6, dim=16, n_layers=1,
                          n_heads=2, timesteps=8)
    x = np.random.randint(0, 11, (4, 6))
    out = m.forward(x, training=True)
    assert np.asarray(out).shape == (4, 6, 12)
    loss = MaskedSparseCrossentropy(m, from_logits=True)
    val = loss(x, np.asarray(out))
    assert np.isfinite(val)
    g = loss.gradient(x, np.asarray(out))
    assert g.shape == (4, 6, 12)
    s = m.sample(seq_len=6, steps=2)
    assert s.shape[1] == 6
