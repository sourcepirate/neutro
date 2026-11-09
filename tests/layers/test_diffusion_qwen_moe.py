import numpy as np
from neutro.autograd import Tensor
from neutro.layers.core.swiglu_mlp import SwiGLUMLP
from neutro.layers.core.moe import MoELayer
from neutro.layers.normalization.qknorm import QKNorm
from neutro.layers.diffusion.time_mlp import TimeMLPEmbedding
from neutro.layers.diffusion.adaln import AdaLNZero
from neutro.layers.diffusion.dit_block import DiTBlock
from neutro.layers.diffusion.discrete_scheduler import DiscreteDiffusionScheduler
from neutro.layers.transformer.qwen_block import QwenBlock
from neutro.layers.transformer.moe_block import MoETransformerBlock


def test_swiglu_shapes():
    m = SwiGLUMLP(8, 16)
    m.build((None, 4, 8))
    out = m(Tensor(np.random.randn(2, 4, 8)))
    assert out.shape == (2, 4, 8)


def test_moe_topk_and_aux():
    m = MoELayer(num_experts=4, top_k=2, expert_units=8)
    m.build((None, 3, 6))
    out = m(Tensor(np.random.randn(2, 3, 6)))
    assert out.shape == (2, 3, 6)
    assert m.aux_loss() >= 0


def test_moe_swiglu_expert():
    m = MoELayer(num_experts=2, top_k=1, expert_type='swiglu',
                 expert_hidden_dim=8)
    m.build((None, 3, 6))
    out = m(Tensor(np.random.randn(1, 3, 6)))
    assert out.shape == (1, 3, 6)


def test_qknorm_identity_scale():
    n = QKNorm()
    n.build((None, 4, 8))
    x = Tensor(np.random.randn(2, 4, 8))
    assert n(x, training=False).shape == (2, 4, 8)


def test_adaln_zero_init_identity():
    a = AdaLNZero()
    a.build([(None, 4, 8), (None, 6)])
    x = Tensor(np.random.randn(2, 4, 8))
    c = Tensor(np.random.randn(2, 6))
    # gate/scale/shift are zero-init: normed path differs, but gate outputs ~0
    g = a.gate_proj(c)
    assert np.allclose(np.asarray(g.data), 0, atol=1e-9)


def test_dit_block_roundtrip():
    b = DiTBlock(16, 2)
    b.build([(None, 4, 16), (None, 16)])
    h = Tensor(np.random.randn(1, 4, 16))
    c = Tensor(np.random.randn(1, 16))
    out = b([h, c])
    assert out.shape == (1, 4, 16)
    grad = b.backward(np.ones(out.shape))
    assert isinstance(grad, list) and len(grad) == 2


def test_scheduler_masks():
    s = DiscreteDiffusionScheduler(timesteps=10, mask_id=99)
    xt, mask = s.q_sample(np.array([[1, 2, 3]]), 9)
    assert xt.shape == (1, 3) and mask.shape == (1, 3)
    assert (xt[mask] == 99).all()


def test_qwen_and_moe_blocks():
    q = QwenBlock(16, 2)
    q.build((None, 4, 16))
    assert q(Tensor(np.random.randn(1, 4, 16))).shape == (1, 4, 16)
    mb = MoETransformerBlock(16, 2, n_experts=4, top_k=2)
    mb.build((None, 4, 16))
    assert mb(Tensor(np.random.randn(1, 4, 16))).shape == (1, 4, 16)
    assert mb.aux_loss() >= 0


def test_time_mlp_shape():
    t = TimeMLPEmbedding(16)
    t.build((4,))
    out = t(np.array([1, 2, 3, 4]))
    assert out.shape == (4, 16)
