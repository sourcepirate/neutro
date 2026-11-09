# Tutorial 06 — Tokenization

**Time:** about 15 minutes. **Prerequisites:** [Tutorial 04](04-tiny-transformer-lm.md).

Language models do not see text. They see integer token ids. This lesson trains
a tiny byte-level BPE tokenizer and walks through encoding, decoding, special
tokens, and saving/loading.

Run the companion script:

```bash
python examples/06_tokenization_tour.py
```

Expected output, approximately:

```text
BPETokenizer
  ids:       [271, 115, 259]
  decoded:   the cat sat
  vocab:     294

RegexTokenizer
  ids:       [257, 32, 261, 32, 260]
  decoded:   the cat sat
  vocab:     272

Special token
  ids:       [257, 32, 261, 32, 300, 32, 257, 32, 263]
  decoded:   the cat <|endoftext|> the dog

Unicode round-trip
  ids:       [104, 101, 108, 108, 111, 32, 240, 159, 140, 141]
  decoded:   hello 🌍

Save/load
  same ids:  True
```

The exact ids depend on the training text and vocabulary size. The round-trip
and shape behavior do not.

---

## 1. Why bytes?

A word-level tokenizer needs an `<unk>` token for words it has never seen.
A **byte-level** tokenizer starts from the 256 possible bytes, so every UTF-8
string can be represented. No unknown tokens.

The trade-off: a rare character may need several byte tokens.

```python
ids = regex.encode("hello 🌍")
print(ids)
print(regex.decode(ids))
```

The emoji becomes several byte ids, then decodes back exactly.

---

## 2. BPE in three steps

1. **Pre-tokenize** — split text into chunks with a regex (words, punctuation,
   whitespace).
2. **Byte-encode** — turn each chunk into UTF-8 bytes.
3. **Merge** — repeatedly replace the most frequent adjacent pair of ids with a
   new id, until `vocab_size` is reached.

Encoding applies the learned merges greedily, using the pair with the lowest
learned rank first.

---

## 3. Train a plain BPE tokenizer

```python
from neutro.tokenizers import BPETokenizer

bpe = BPETokenizer()
bpe.train(TEXT, vocab_size=300)

ids = bpe.encode("the cat sat")
print(ids)
print(bpe.decode(ids))
```

`train` takes a **string**, not a list of strings. `vocab_size` must be at least
256 because the byte vocabulary is always present.

`len(bpe.encoder)` may be less than `vocab_size` when the training text runs out
of pairs to merge.

---

## 4. Regex BPE

`RegexTokenizer` adds GPT-style pre-tokenization and special tokens:

```python
from neutro.tokenizers import RegexTokenizer

tok = RegexTokenizer()
tok.train(TEXT, vocab_size=300)

ids = tok.encode("the cat sat")
print(tok.decode(ids))
```

The default pattern is `GPT4_SPLIT_PATTERN`. You can pass a custom regex:

```python
tok = RegexTokenizer(pattern=r"\w+|[^\w\s]+|\s+")
```

Pre-tokenization matters: it stops merges from crossing word or punctuation
boundaries in ways you did not intend.

---

## 5. Special tokens

Reserve ids for markers such as end-of-text:

```python
tok.register_special_tokens({"<|endoftext|>": 300})

ids = tok.encode("the cat <|endoftext|> the dog", allowed_special="all")
print(ids)
print(tok.decode(ids))
```

`allowed_special` controls how special-token strings are treated:

- `"none"` (default) — treat the marker as ordinary text.
- `"all"` — encode every registered marker as its reserved id.
- an iterable of keys — allow only those markers.

`decode` maps the reserved id back to its string.

---

## 6. Save and load

```python
tok.save("tokenizer")

loaded = RegexTokenizer()
loaded.load("tokenizer")
```

This writes/reads `tokenizer.json` with the pattern, encoder, and special
tokens. `tok.vocab_size` includes byte/merge tokens and special tokens.

Use a temporary directory in tests so you do not leave artifacts behind:

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    prefix = str(Path(tmp) / "tokenizer")
    tok.save(prefix)
    ...
```

---

## 7. Connect it to a model

A typical language-model pipeline:

```text
text -> tokenizer.encode -> (B, S) ids
     -> Embedding(vocab_size, dim)
     -> TransformerBlock
     -> Dense(vocab_size)
     -> logits
     -> tokenizer.decode(sampled ids)
```

The model's `vocab_size` must match the tokenizer's vocabulary size (including
special tokens).

---

## 8. Your turn

1. Train on a larger text file. Does the tokenizer produce fewer tokens per
   sentence?
2. Try `vocab_size=260` and `vocab_size=400`. How does the encoding length
   change?
3. Encode a sentence with emoji and accented characters. Does it round-trip?
4. Register two special tokens. What happens if they share an id?
5. What happens if you call `encode(..., allowed_special="all")` before
   registering any special tokens?

<details>
<summary>Answers</summary>

1. More merges usually means fewer tokens per sentence, until the useful pairs
   are exhausted.
2. A smaller vocabulary produces longer id sequences; a larger one produces
   shorter sequences but a larger embedding matrix.
3. Yes; byte-level BPE has no `<unk>` token.
4. The decoder map is inverted from the special-token dict, so duplicate ids
   make decoding ambiguous. Use unique ids.
5. There is nothing to match, so the text is encoded normally.

</details>

---

## What to read next

- [Tokenizer reference](../tokenizers/tokenizers.md)
- [Embedding layer](../layers/embedding/embedding.md)
- [Tutorial 04 — Tiny transformer LM](04-tiny-transformer-lm.md)
- [Tutorial 07 — LoRA fine-tuning](07-lora-finetuning.md) *(next)*
