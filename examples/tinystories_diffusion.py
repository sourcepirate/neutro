"""Masked diffusion LM toy example (LLaDA/MDLM-style).

Trains a tiny DiT-backed MaskedDiffusionLM on synthetic repetitive text,
then unmasks a prompt. CPU-friendly: vocab ~40, seq 16, 1-2 layers.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models.language.diffusion_lm import MaskedDiffusionLM
from neutro.losses.masked_sparse_ce import MaskedSparseCrossentropy
from neutro.optimizers import AdamW
from neutro.tokenizers import RegexTokenizer


def toy_corpus(n=2000):
    subjects = ["the cat", "a dog", "the bird", "a fish"]
    verbs = ["sat", "ran", "sang", "swam"]
    places = ["on the mat", "in the park", "by the lake", "at home"]
    return [f"{s} {v} {p}. " for _ in range(n)
            for s, v, p in [(np.random.choice(subjects),
                             np.random.choice(verbs),
                             np.random.choice(places))]][:n]


def main():
    texts = toy_corpus(500)
    blob = "".join(texts)
    print("Training tokenizer...")
    tok = RegexTokenizer()
    tok.train(blob[:20000], vocab_size=256, verbose=False)
    print(f"vocab={tok.vocab_size} mask_id={tok.vocab_size - 1}")

    seq_len = 16
    ids = np.array(tok.encode(blob), dtype=np.int64)
    # Reserve last id as [MASK]: clip training ids away from it.
    ids = np.clip(ids, 0, tok.vocab_size - 2)
    mask_id = tok.vocab_size - 1
    xs = np.stack([ids[i:i + seq_len] for i in range(0, len(ids) - seq_len, 4)])
    print(f"data {xs.shape}")

    model = MaskedDiffusionLM(vocab_size=tok.vocab_size, seq_len=seq_len,
                              dim=32, n_layers=1, n_heads=2,
                              timesteps=32, mask_id=mask_id)
    model.compile(optimizer=AdamW(learning_rate=0.002),
                  loss=MaskedSparseCrossentropy(model, from_logits=True))
    model.summary()
    model.fit(xs[:400], xs[:400], epochs=2, batch_size=32)

    prompt = np.array([tok.encode("the cat ")[:6]], dtype=np.int64)
    out = model.sample(prefix=prompt, seq_len=seq_len, steps=8)
    print("prompt:", tok.decode(prompt[0].tolist()))
    print("sample:", tok.decode([int(i) for i in out[0] if i < tok.vocab_size - 1]))


if __name__ == "__main__":
    main()
