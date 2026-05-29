import numpy as np
import pytest
from neutro.models.language.gpt import GPT1, GPT2
from neutro.models.language.qwen import Qwen1Tiny, Qwen2Tiny, Qwen3Tiny
from neutro.models.language.deepseek import DeepSeekV1Tiny, DeepSeekV2Tiny, DeepSeekV3Tiny, DeepSeekV4Tiny

@pytest.mark.parametrize("model_fn", [GPT1, GPT2])
def test_gpt_smoke(model_fn):
    vocab_size = 100
    seq_len = 10
    model = model_fn(vocab_size, seq_len, dim=32, n_heads=4, n_layers=1)
    x = np.random.randint(0, vocab_size, (1, seq_len))
    y = model.predict(x)
    assert y.shape == (1, seq_len, vocab_size)

@pytest.mark.parametrize("model_fn", [Qwen1Tiny, Qwen2Tiny, Qwen3Tiny])
def test_qwen_smoke(model_fn):
    vocab_size = 100
    seq_len = 10
    model = model_fn(vocab_size, seq_len, dim=32, n_layers=1, n_heads=4)
    x = np.random.randint(0, vocab_size, (1, seq_len))
    y = model.predict(x)
    assert y.shape == (1, seq_len, vocab_size)

@pytest.mark.parametrize("model_fn", [DeepSeekV1Tiny, DeepSeekV2Tiny, DeepSeekV3Tiny, DeepSeekV4Tiny])
def test_deepseek_smoke(model_fn):
    vocab_size = 100
    seq_len = 10
    model = model_fn(vocab_size, seq_len, dim=32, n_layers=1, n_heads=4)
    x = np.random.randint(0, vocab_size, (1, seq_len))
    y = model.predict(x)
    assert y.shape == (1, seq_len, vocab_size)

def test_model_generate():
    vocab_size = 100
    seq_len = 10
    model = GPT2(vocab_size, seq_len, dim=32, n_heads=4, n_layers=1)
    start_tokens = np.random.randint(0, vocab_size, (1, 5))
    generated = model.generate(start_tokens, max_new_tokens=5)
    assert generated.shape == (1, 10)
