import numpy as np
import pytest
from neutro.layers.attention.paged_attention import PagedKVCache, PagedAttention
from neutro.layers.attention.kv_cache import KVCache
from neutro.layers.attention.mha import MultiHeadAttention
from neutro.layers.base import Layer
from neutro.models import Sequential, Model


# ─── PagedKVCache Tests ────────────────────────────────────────────────

class TestPagedKVCache:

    def test_basic_update_and_reset(self):
        cache = PagedKVCache(num_blocks=8, block_size=4)
        k = np.random.randn(1, 2, 1, 8)
        v = np.random.randn(1, 2, 1, 8)
        cache.update(k, v, layer_id=0)
        assert cache.get_num_tokens(0) == 1
        assert len(cache.block_tables[0]) == 1
        assert cache.block_fill[cache.block_tables[0][0]] == 1

        cache.update(k, v, layer_id=0)
        assert cache.get_num_tokens(0) == 2
        assert len(cache.block_tables[0]) == 1
        assert cache.block_fill[cache.block_tables[0][0]] == 2

        cache.reset()
        assert len(cache.block_tables) == 0
        assert len(cache.free_blocks) == cache.num_blocks
        assert cache.kv_blocks is None

    def test_block_allocation(self):
        cache = PagedKVCache(num_blocks=8, block_size=4)
        k = np.random.randn(1, 2, 1, 8)
        v = np.random.randn(1, 2, 1, 8)

        for _ in range(9):
            cache.update(k, v, layer_id=0)

        assert cache.get_num_tokens(0) == 9
        assert len(cache.block_tables[0]) == 3
        assert cache.block_fill[cache.block_tables[0][0]] == 4
        assert cache.block_fill[cache.block_tables[0][1]] == 4
        assert cache.block_fill[cache.block_tables[0][2]] == 1

    def test_multi_layer(self):
        cache = PagedKVCache(num_blocks=8, block_size=4)
        k = np.random.randn(1, 2, 1, 8)
        v = np.random.randn(1, 2, 1, 8)

        cache.update(k, v, layer_id=0)
        cache.update(k, v, layer_id=1)

        assert cache.get_num_tokens(0) == 1
        assert cache.get_num_tokens(1) == 1
        assert len(cache.block_tables[0]) == 1
        assert len(cache.block_tables[1]) == 1

    def test_large_prefill(self):
        cache = PagedKVCache(num_blocks=16, block_size=4)
        k = np.random.randn(1, 2, 10, 8)
        v = np.random.randn(1, 2, 10, 8)
        cache.update(k, v, layer_id=0)

        assert cache.get_num_tokens(0) == 10
        assert len(cache.block_tables[0]) == 3

    def test_reset_reuse(self):
        cache = PagedKVCache(num_blocks=4, block_size=4)
        k = np.random.randn(1, 2, 4, 8)
        v = np.random.randn(1, 2, 4, 8)
        cache.update(k, v, layer_id=0)
        assert len(cache.free_blocks) == 3

        cache.reset()
        cache.update(k, v, layer_id=0)
        assert cache.get_num_tokens(0) == 4
        assert len(cache.free_blocks) == 3

    def test_no_free_blocks(self):
        cache = PagedKVCache(num_blocks=1, block_size=4)
        k = np.random.randn(1, 2, 4, 8)
        v = np.random.randn(1, 2, 4, 8)
        cache.update(k, v, layer_id=0)
        assert cache.get_num_tokens(0) == 4

        with pytest.raises(IndexError):
            k2 = np.random.randn(1, 2, 1, 8)
            v2 = np.random.randn(1, 2, 1, 8)
            cache.update(k2, v2, layer_id=0)

    def test_get_block_table(self):
        cache = PagedKVCache(num_blocks=8, block_size=4)
        k = np.random.randn(1, 2, 5, 8)
        v = np.random.randn(1, 2, 5, 8)
        cache.update(k, v, layer_id=0)

        bt, bf = cache.get_block_table(layer_id=0)
        assert len(bt) == 2
        assert bf[bt[0]] == 4
        assert bf[bt[1]] == 1

        bt_empty, bf_empty = cache.get_block_table(layer_id=99)
        assert bt_empty == []

    def test_kv_content_preserved(self):
        cache = PagedKVCache(num_blocks=2, block_size=4)
        k = np.arange(1 * 2 * 4 * 8, dtype=np.float32).reshape(1, 2, 4, 8)
        v = np.arange(1 * 2 * 4 * 8, dtype=np.float32).reshape(1, 2, 4, 8) * 10
        cache.update(k, v, layer_id=0)

        bt, bf = cache.get_block_table(0)
        phys = bt[0]
        fill = int(bf[phys])
        stored_k = cache.kv_blocks[phys, 0, :, :fill, :]
        np.testing.assert_allclose(stored_k, k[0])
        stored_v = cache.kv_blocks[phys, 1, :, :fill, :]
        np.testing.assert_allclose(stored_v, v[0])


# ─── PagedAttention Tests ──────────────────────────────────────────────

class TestPagedAttention:

    def test_build(self):
        layer = PagedAttention(num_heads=4, key_dim=16)
        layer.build((2, 8, 16))
        assert layer.built
        for p in ['Wq', 'Wk', 'Wv', 'Wo']:
            assert p in layer.params

    def test_forward_no_cache(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        x = np.random.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 16)

    def test_backward_shape(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        x = np.random.randn(2, 5, 16)
        out = layer(x)
        grad = np.random.randn(2, 5, 16)
        dx = layer.backward(grad)
        assert dx.shape == (2, 5, 16)
        for p in ['Wq', 'Wk', 'Wv', 'Wo']:
            assert p in layer.grads

    def test_forward_parity_with_mha(self):
        batch, seq_len, embed_dim = 2, 8, 16
        num_heads = 2
        key_dim = 16
        x = np.random.randn(batch, seq_len, embed_dim)

        mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        pa = PagedAttention(num_heads=num_heads, key_dim=key_dim)

        mha.build(x.shape)
        pa.build(x.shape)

        pa.params['Wq'] = mha.params['Wq'].copy()
        pa.params['Wk'] = mha.params['Wk'].copy()
        pa.params['Wv'] = mha.params['Wv'].copy()
        pa.params['Wo'] = mha.params['Wo'].copy()

        out_mha = mha.forward(x)
        out_pa = pa.forward(x)

        np.testing.assert_allclose(out_mha, out_pa, atol=1e-5)

    def test_gradient_parity_with_mha(self):
        batch, seq_len, embed_dim = 2, 6, 12
        num_heads = 2
        key_dim = 12
        x = np.random.randn(batch, seq_len, embed_dim)
        grad_out = np.random.randn(batch, seq_len, embed_dim)

        mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        pa = PagedAttention(num_heads=num_heads, key_dim=key_dim)

        mha.build(x.shape)
        pa.build(x.shape)

        pa.params['Wq'] = mha.params['Wq'].copy()
        pa.params['Wk'] = mha.params['Wk'].copy()
        pa.params['Wv'] = mha.params['Wv'].copy()
        pa.params['Wo'] = mha.params['Wo'].copy()

        mha.forward(x)
        pa.forward(x)

        dx_mha = mha.backward(grad_out)
        dx_pa = pa.backward(grad_out)

        np.testing.assert_allclose(dx_mha, dx_pa, atol=1e-5)
        for p in ['Wq', 'Wk', 'Wv', 'Wo']:
            np.testing.assert_allclose(
                mha.grads[p], pa.grads[p], atol=1e-5,
                err_msg=f"Gradient mismatch for {p}"
            )

    def test_with_paged_cache_prefill_and_decode(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 5, 16))

        cache = PagedKVCache(num_blocks=16, block_size=4)

        x_prefill = np.random.randn(1, 5, 16)
        out_prefill = layer(x_prefill, kv_cache=cache, layer_id=0)
        assert out_prefill.shape == (1, 5, 16)
        assert cache.get_num_tokens(0) == 5

        x_decode = np.random.randn(1, 1, 16)
        out_decode = layer(x_decode, kv_cache=cache, layer_id=0)
        assert out_decode.shape == (1, 1, 16)
        assert cache.get_num_tokens(0) == 6

    def test_paged_cache_multiple_decode_steps(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 1, 16))

        cache = PagedKVCache(num_blocks=16, block_size=4)

        for step in range(6):
            x = np.random.randn(1, 1, 16)
            out = layer(x, kv_cache=cache, layer_id=0)
            assert out.shape == (1, 1, 16)
            expected_tokens = step + 1
            assert cache.get_num_tokens(0) == expected_tokens

    def test_causal_mask(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 4, 16))

        x = np.random.randn(1, 4, 16)
        mask = np.triu(np.ones((4, 4)), k=1)

        out = layer(x, mask=mask)
        assert out.shape == (1, 4, 16)

        layer2 = PagedAttention(num_heads=2, key_dim=16)
        layer2.build((1, 4, 16))
        layer2.params['Wq'] = layer.params['Wq'].copy()
        layer2.params['Wk'] = layer.params['Wk'].copy()
        layer2.params['Wv'] = layer.params['Wv'].copy()
        layer2.params['Wo'] = layer.params['Wo'].copy()

        out_nomask = layer2(x)
        assert not np.allclose(out, out_nomask)

    def test_causal_mask_with_paged_cache(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 5, 16))

        cache = PagedKVCache(num_blocks=16, block_size=4)
        x = np.random.randn(1, 5, 16)
        mask = np.triu(np.ones((5, 5)), k=1)
        out = layer(x, mask=mask, kv_cache=cache, layer_id=0)
        assert out.shape == (1, 5, 16)

        x2 = np.random.randn(1, 1, 16)
        mask2 = np.zeros((1, 6))
        out2 = layer(x2, mask=mask2, kv_cache=cache, layer_id=0)
        assert out2.shape == (1, 1, 16)

    def test_rope(self):
        layer = PagedAttention(num_heads=2, key_dim=16, use_rope=True)
        x = np.random.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 16)

    def test_rope_with_paged_cache(self):
        layer = PagedAttention(num_heads=2, key_dim=16, use_rope=True)
        layer.build((1, 1, 16))

        cache = PagedKVCache(num_blocks=16, block_size=4)

        for step in range(4):
            x = np.random.randn(1, 1, 16)
            out = layer(x, kv_cache=cache, layer_id=0)
            assert out.shape == (1, 1, 16)

    def test_rope_gradient_parity(self):
        from neutro.layers.attention.flash_attention import FlashAttention
        batch, seq_len, embed_dim = 1, 5, 16
        num_heads = 2
        key_dim = 16
        x = np.random.randn(batch, seq_len, embed_dim)
        grad_out = np.random.randn(batch, seq_len, embed_dim)

        fa = FlashAttention(num_heads=num_heads, key_dim=key_dim, use_rope=True)
        pa = PagedAttention(num_heads=num_heads, key_dim=key_dim, use_rope=True)

        fa.build(x.shape)
        pa.build(x.shape)

        pa.params['Wq'] = fa.params['Wq'].copy()
        pa.params['Wk'] = fa.params['Wk'].copy()
        pa.params['Wv'] = fa.params['Wv'].copy()
        pa.params['Wo'] = fa.params['Wo'].copy()

        fa.forward(x)
        pa.forward(x)

        dx_fa = fa.backward(grad_out)
        dx_pa = pa.backward(grad_out)

        np.testing.assert_allclose(dx_fa, dx_pa, atol=1e-4)
        for p in ['Wq', 'Wk', 'Wv', 'Wo']:
            np.testing.assert_allclose(
                fa.grads[p], pa.grads[p], atol=1e-4,
                err_msg=f"Gradient mismatch for {p} with RoPE"
            )

    def test_block_size_one(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 1, 16))
        cache = PagedKVCache(num_blocks=16, block_size=1)

        for step in range(5):
            x = np.random.randn(1, 1, 16)
            out = layer(x, kv_cache=cache, layer_id=0)
            assert out.shape == (1, 1, 16)
            assert cache.get_num_tokens(0) == step + 1
            assert len(cache.block_tables[0]) == step + 1

    def test_single_head(self):
        layer = PagedAttention(num_heads=1, key_dim=8)
        x = np.random.randn(2, 5, 8)
        out = layer(x)
        assert out.shape == (2, 5, 8)
        grad = np.random.randn(2, 5, 8)
        dx = layer.backward(grad)
        assert dx.shape == (2, 5, 8)

    def test_no_cache_via_call(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        x = np.random.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 16)

    def test_with_regular_kv_cache(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 1, 16))
        cache = KVCache()

        x = np.random.randn(1, 1, 16)
        out = layer(x, kv_cache=cache, layer_id=0)
        assert out.shape == (1, 1, 16)
        assert cache.k_cache[0].shape == (1, 2, 1, 8)

        out2 = layer(x, kv_cache=cache, layer_id=0)
        assert out2.shape == (1, 1, 16)
        assert cache.k_cache[0].shape == (1, 2, 2, 8)

    def test_gradient_with_regular_cache(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 5, 16))
        cache = KVCache()

        x1 = np.random.randn(1, 1, 16)
        out1 = layer(x1, kv_cache=cache, layer_id=0)
        x2 = np.random.randn(1, 1, 16)
        out2 = layer(x2, kv_cache=cache, layer_id=0)

        grad = np.random.randn(1, 1, 16)
        dx = layer.backward(grad)
        assert dx.shape == (1, 1, 16)
        for p in ['Wq', 'Wk', 'Wv', 'Wo']:
            assert p in layer.grads

    def test_compute_output_shape(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        shape = layer.compute_output_shape((2, 5, 16))
        assert shape == (2, 5, 16)

    def test_sublayers_empty(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        assert layer.sublayers == []

    def test_count_params(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 5, 16))
        assert layer.count_params() > 0

    def test_different_block_sizes(self):
        for block_size in [1, 2, 8, 16]:
            layer = PagedAttention(num_heads=2, key_dim=16)
            layer.build((1, 1, 16))
            cache = PagedKVCache(num_blocks=16, block_size=block_size)

            x = np.random.randn(1, 1, 16)
            out = layer(x, kv_cache=cache, layer_id=0)
            assert out.shape == (1, 1, 16)

    def test_batch_two_parity(self):
        batch, seq_len, embed_dim = 3, 5, 16
        num_heads = 2
        key_dim = 16
        x = np.random.randn(batch, seq_len, embed_dim)

        mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        pa = PagedAttention(num_heads=num_heads, key_dim=key_dim)
        mha.build(x.shape)
        pa.build(x.shape)

        pa.params['Wq'] = mha.params['Wq'].copy()
        pa.params['Wk'] = mha.params['Wk'].copy()
        pa.params['Wv'] = mha.params['Wv'].copy()
        pa.params['Wo'] = mha.params['Wo'].copy()

        out_mha = mha.forward(x)
        out_pa = pa.forward(x)
        np.testing.assert_allclose(out_mha, out_pa, atol=1e-5)

    def test_model_integration_sequential(self):
        vocab_size = 20
        seq_len = 8
        embed_dim = 16
        num_heads = 2

        class PagedModel(Model):
            def __init__(self, vocab_size, seq_len, embed_dim, num_heads):
                super().__init__()
                from neutro.layers import Embedding, Dense
                self.embed = Embedding(vocab_size, embed_dim)
                self.attn = PagedAttention(num_heads=num_heads, key_dim=embed_dim)
                self.out = Dense(vocab_size)
                self.seq_len = seq_len

            def build(self, input_shape):
                self.embed.build(input_shape)
                self.attn.build(self.embed.compute_output_shape(input_shape))
                self.out.build(self.attn.compute_output_shape(
                    self.embed.compute_output_shape(input_shape)))
                self.built = True

            def forward(self, x, training=False, kv_cache=None):
                x = self.embed(x)
                if kv_cache is not None and isinstance(kv_cache, PagedKVCache):
                    x = self.attn(x, kv_cache=kv_cache, layer_id=0)
                else:
                    x = self.attn(x)
                return self.out(x)

        model = PagedModel(vocab_size, seq_len, embed_dim, num_heads)
        model.build((1, seq_len))

        input_tokens = np.random.randint(0, vocab_size, (1, 3))
        logits = model(input_tokens)
        assert logits.shape == (1, 3, vocab_size)

        cache = PagedKVCache(num_blocks=16, block_size=4)
        logits_cached = model.forward(input_tokens, kv_cache=cache)
        assert logits_cached.shape == (1, 3, vocab_size)
        assert cache.get_num_tokens(0) == 3

        new_token = np.random.randint(0, vocab_size, (1, 1))
        logits_next = model.forward(new_token, kv_cache=cache)
        assert logits_next.shape == (1, 1, vocab_size)
        assert cache.get_num_tokens(0) == 4

    def test_model_gradients_with_paged_cache(self):
        vocab_size = 10
        embed_dim = 8
        num_heads = 1

        class TrainablePagedModel(Model):
            def __init__(self, vocab_size, embed_dim, num_heads):
                super().__init__()
                from neutro.layers import Embedding, Dense
                self.embed = Embedding(vocab_size, embed_dim)
                self.attn = PagedAttention(num_heads=num_heads, key_dim=embed_dim)
                self.out = Dense(vocab_size)
                self.layers = [self.embed, self.attn, self.out]

            def build(self, input_shape):
                self.embed.build(input_shape)
                attn_shape = self.embed.compute_output_shape(input_shape)
                self.attn.build(attn_shape)
                out_shape = self.attn.compute_output_shape(attn_shape)
                self.out.build(out_shape)
                self.built = True

            def forward(self, x, training=False, kv_cache=None):
                x = self.embed(x)
                x = self.attn(x)
                return self.out(x)

        model = TrainablePagedModel(vocab_size, embed_dim, num_heads)
        x = np.random.randint(0, vocab_size, (2, 4))
        y = np.random.randint(0, vocab_size, (2, 4))

        from neutro.optimizers import SGD
        from neutro.losses import SparseCategoricalCrossentropy
        model.compile(optimizer=SGD(0.01), loss='sparse_categorical_crossentropy')
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)

        all_layers = model._get_all_layers()
        attn = [l for l in all_layers if isinstance(l, PagedAttention)][0]
        for p in ['Wq', 'Wk', 'Wv', 'Wo']:
            assert p in attn.grads
            assert not np.allclose(attn.grads[p], 0)

    def test_forward_then_backward_multiple_calls(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        x1 = np.random.randn(2, 4, 16)
        out1 = layer(x1)
        grad1 = np.random.randn(2, 4, 16)
        dx1 = layer.backward(grad1)
        assert dx1.shape == (2, 4, 16)

        x2 = np.random.randn(2, 4, 16)
        out2 = layer(x2)
        grad2 = np.random.randn(2, 4, 16)
        dx2 = layer.backward(grad2)
        assert dx2.shape == (2, 4, 16)

    def test_forward_preserves_input(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        x = np.random.randn(2, 5, 16)
        x_copy = x.copy()
        layer(x)
        np.testing.assert_allclose(x, x_copy)

    def test_paged_cache_gradient_backward(self):
        layer = PagedAttention(num_heads=2, key_dim=16)
        layer.build((1, 5, 16))
        cache = PagedKVCache(num_blocks=16, block_size=4)

        x = np.random.randn(1, 5, 16)
        out = layer(x, kv_cache=cache, layer_id=0)
        grad = np.random.randn(1, 5, 16)
        dx = layer.backward(grad)
        assert dx.shape == (1, 5, 16)
        for p in ['Wq', 'Wk', 'Wv', 'Wo']:
            assert p in layer.grads
            assert not np.allclose(layer.grads[p], 0)

    def test_paged_cache_gradient_parity_with_mha(self):
        batch, seq_len, embed_dim = 1, 5, 16
        num_heads = 2
        key_dim = 16
        x = np.random.randn(batch, seq_len, embed_dim)
        grad_out = np.random.randn(batch, seq_len, embed_dim)

        mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        pa = PagedAttention(num_heads=num_heads, key_dim=key_dim)
        mha.build(x.shape)
        pa.build(x.shape)

        pa.params['Wq'] = mha.params['Wq'].copy()
        pa.params['Wk'] = mha.params['Wk'].copy()
        pa.params['Wv'] = mha.params['Wv'].copy()
        pa.params['Wo'] = mha.params['Wo'].copy()

        cache = PagedKVCache(num_blocks=16, block_size=4)

        mha.forward(x)
        pa.forward(x, kv_cache=cache, layer_id=0)

        dx_mha = mha.backward(grad_out)
        dx_pa = pa.backward(grad_out)

        np.testing.assert_allclose(dx_mha, dx_pa, atol=1e-5)
        for p in ['Wq', 'Wk', 'Wv', 'Wo']:
            np.testing.assert_allclose(
                mha.grads[p], pa.grads[p], atol=1e-5,
                err_msg=f"Gradient mismatch for {p} with paged cache"
            )
