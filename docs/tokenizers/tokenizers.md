# Tokenizers

## Theory

Tokenizers convert raw text into integer token IDs for model input. `neutro` implements byte-level BPE (Byte-Pair Encoding), the algorithm used by GPT-2/4 and Llama.

### BPE Tokenizer — `neutro/tokenizers/bpe.py`

Byte-Pair Encoding works in three stages:

1. **Pre-tokenization**: Split text into words using regex (GPT-2 style: `'s | 't | 're | 've | 'm | 'll | 'n | \p{L}+ | \p{N}+ | [^\s\p{L}\p{N}]+`).
2. **Byte encoding**: Convert each character to its byte representation to handle all Unicode seamlessly (GPT-2 trick).
3. **Merge learning**: Repeatedly replace the most frequent pair of adjacent tokens with a new merged token until a vocabulary size is reached.

Encoding a new text:
```python
def encode(self, text):
    tokens = self._pre_tokenize(text)
    for pair, new_id in sorted(self.merges):
        tokens = self._merge_pair(tokens, pair, new_id)
    return tokens
```

### RegexTokenizer — `neutro/tokenizers/bpe.py`

Extends BPE with:
- Regex-based pre-tokenization (configurable pattern).
- Special token handling (`<|endoftext|>`, etc.).
- `encode` and `decode` methods consistent with the Keras style.

### TikToken Compatibility — `neutro/tokenizers/tiktoken_compat.py`

Provides a `TikTokenTokenizer` that can load OpenAI's TikToken BPE ranks (from `cl100k_base`, `gpt2`, etc.) and use them directly. This bridges `neutro` with OpenAI's tokenization system.

```python
def load_tiktoken_bpe(bpe_file):
    # Reads the BPE merge file in TikToken format
    with open(bpe_file, 'rb') as f:
        return pickle.load(f)
```

## Usage Example

```python
from neutro.tokenizers import RegexTokenizer, TikTokenTokenizer

tokenizer = RegexTokenizer()
tokenizer.train(["hello world", "hello there"])
tokens = tokenizer.encode("hello world")
text = tokenizer.decode(tokens)
```

## References

- Sennrich, R., Haddow, B., & Birch, A. (2016). **Neural Machine Translation of Rare Words with Subword Units**. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- GPT-2 Tokenizer: Radford, A., et al. (2019). **Language Models are Unsupervised Multitask Learners**.
