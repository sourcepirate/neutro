# Tokenizers

Tokenizers convert raw text into integer token ids for a model.

Neutro implements byte-level BPE (Byte-Pair Encoding), the algorithm behind
GPT-2/4 and many Llama-style tokenizers.

Source: `neutro/tokenizers/bpe.py` and `neutro/tokenizers/tiktoken_compat.py`.

---

## Byte-level BPE

The algorithm has three stages:

1. **Pre-tokenization** — split text into chunks with a regex (words,
   punctuation, whitespace, …).
2. **Byte encoding** — convert each chunk to UTF-8 bytes. This handles any
   Unicode text without an unknown-token problem.
3. **Merge learning** — repeatedly replace the most frequent adjacent pair of
   token ids with a new id, until the vocabulary reaches the requested size.

Encoding applies the learned merges greedily, always merging the pair with the
lowest learned rank first.

---

## `BPETokenizer`

Plain byte-level BPE without regex splitting:

```python
from neutro.tokenizers import BPETokenizer

tok = BPETokenizer
tok.train("hello world, hello there", vocab_size=300)

ids = tok.encode("hello world")
text = tok.decode(ids)
print(ids[:10]) # integer ids
print(text) # "hello world"
```

`train` takes a **string**, not a list of strings. `vocab_size` must be at least
256 because the base byte vocabulary is always present.

---

## `RegexTokenizer`

Adds GPT-style regex pre-tokenization and special tokens:

```python
from neutro.tokenizers import RegexTokenizer

tok = RegexTokenizer()
tok.train("hello world. hello there!", vocab_size=300)

tok.register_special_tokens({"<|endoftext|>": 300})
ids = tok.encode("hello world", allowed_special="all")
print(tok.decode(ids))
```

`allowed_special` controls how special-token strings are treated during
encoding: `"none"` (default), `"all"`, or an iterable of allowed keys.

The default pattern is the GPT-4 split pattern (`GPT4_SPLIT_PATTERN`). You can
pass a custom regex via `RegexTokenizer(pattern=...)`.

---

## Saving and loading

```python
tok.save("tokenizer")
tok2 = RegexTokenizer()
tok2.load("tokenizer")
```

This writes/reads `tokenizer.json`, including the pattern, encoder, and special
tokens.

`tok.vocab_size` is the number of byte/merge tokens plus special tokens.

---

## Tiktoken compatibility

`TiktokenCompatibleTokenizer` loads OpenAI's `.tiktoken` rank files:

```python
from neutro.tokenizers import get_gpt2_tokenizer

tok = get_gpt2_tokenizer # downloads r50k_base on first use
ids = tok.encode("hello world")
```

`load_tiktoken_bpe(path_or_url)` parses the base64-encoded token/rank lines and
returns a `bytes -> int` rank map. It does not use pickle.

---

## References

- Sennrich, R., Haddow, B., & Birch, A. (2016). **Neural Machine Translation of Rare Words with Subword Units**. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Radford, A., et al. (2019). **Language Models are Unsupervised Multitask Learners**. [OpenAI](https://openai.com/research/better-language-models/)
- OpenAI tiktoken: <https://github.com/openai/tiktoken>
