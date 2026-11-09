"""A tour of Neutro's byte-level BPE tokenizers.

Run from the repository root::

    python examples/06_tokenization_tour.py

It trains a tiny tokenizer, encodes/decodes text, shows special tokens, and
round-trips Unicode.
"""

import tempfile
from pathlib import Path

from neutro.tokenizers import BPETokenizer, RegexTokenizer

TEXT = (
    "the cat sat on the mat. "
    "the dog sat on the log. "
    "the cat and the dog sat together."
)


def main():
    # Plain byte-level BPE.
    bpe = BPETokenizer()
    bpe.train(TEXT, vocab_size=300)

    ids = bpe.encode("the cat sat")
    print("BPETokenizer")
    print("  ids:      ", ids)
    print("  decoded:  ", bpe.decode(ids))
    print("  vocab:    ", len(bpe.encoder))
    print()

    # Regex BPE: GPT-style pre-tokenization.
    regex = RegexTokenizer()
    regex.train(TEXT, vocab_size=300)

    ids = regex.encode("the cat sat")
    print("RegexTokenizer")
    print("  ids:      ", ids)
    print("  decoded:  ", regex.decode(ids))
    print("  vocab:    ", regex.vocab_size)
    print()

    # Special tokens are encoded as single ids.
    regex.register_special_tokens({"<|endoftext|>": 300})
    special = regex.encode("the cat <|endoftext|> the dog", allowed_special="all")
    print("Special token")
    print("  ids:      ", special)
    print("  decoded:  ", regex.decode(special))
    print()

    # Byte-level BPE can round-trip Unicode without an <unk> token.
    unicode_ids = regex.encode("hello 🌍")
    print("Unicode round-trip")
    print("  ids:      ", unicode_ids)
    print("  decoded:  ", regex.decode(unicode_ids))
    print()

    # Save/load to a temporary prefix.
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "tokenizer")
        regex.save(prefix)
        loaded = RegexTokenizer()
        loaded.load(prefix)
        print("Save/load")
        print("  same ids: ", loaded.encode("the cat sat") == regex.encode("the cat sat"))
    print()
    print("Next: docs/tutorials/06-tokenization.md")


if __name__ == "__main__":
    main()
