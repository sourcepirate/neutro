import base64
import itertools
import json
from collections import Counter

import regex as re


def get_stats(ids):
    """Count the frequency of each adjacent pair of token ids."""
    return Counter(itertools.pairwise(ids))


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` with the new token `idx`."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    """Minimal BPE Tokenizer implementation.

    Educational and "intentionally naive".
    """

    def __init__(self):
        self.encoder = {bytes([i]): i for i in range(256)}
        self.decoder = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def _pair_bytes(self, pair):
        return self.decoder[pair[0]] + self.decoder[pair[1]]

    def _train_loop(self, ids_list, vocab_size, verbose):
        """Run the greedy BPE merge loop across a list of token-id sequences."""
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        for index in range(num_merges):
            stats = Counter(itertools.chain.from_iterable(
                itertools.pairwise(ids) for ids in ids_list
            ))
            if not stats:
                break

            pair = max(stats, key=stats.get)
            token_id = 256 + index
            ids_list = [merge(ids, pair, token_id) for ids in ids_list]

            token = self._pair_bytes(pair)
            self.encoder[token] = token_id
            self.decoder[token_id] = token

            if verbose:
                print(
                    f"merge {index + 1}/{num_merges}: {pair} -> {token_id} "
                    f"had {stats[pair]} occurrences"
                )

    def train(self, text, vocab_size, verbose=False):
        self._train_loop([list(text.encode("utf-8"))], vocab_size, verbose)

    def encode(self, text):
        return self._encode_piece(text.encode("utf-8"))

    def _encode_piece(self, piece_bytes):
        ids = list(piece_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(
                stats,
                key=lambda item: self.encoder.get(self._pair_bytes(item), float("inf")),
            )
            token = self._pair_bytes(pair)
            if token not in self.encoder:
                break
            ids = merge(ids, pair, self.encoder[token])

        return ids

    def decode(self, ids):
        parts = []
        for idx in ids:
            if idx in self.decoder:
                parts.append(self.decoder[idx])
            elif idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")


class RegexTokenizer(BPETokenizer):
    """BPE Tokenizer with Regex splitting (GPT style)."""

    # GPT-4 split pattern
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmtre lve])| \?|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = pattern or self.GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        chunks = self.compiled_pattern.findall(text)
        ids_list = [list(chunk.encode("utf-8")) for chunk in chunks]
        self._train_loop(ids_list, vocab_size, verbose)

    def register_special_tokens(self, special_tokens):
        # special_tokens: dict of str -> int
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def encode(self, text, allowed_special="none"):
        if allowed_special == "none":
            special_tokens = {}
        elif allowed_special == "all":
            special_tokens = self.special_tokens
        else:
            special_tokens = {k: v for k, v in self.special_tokens.items() if k in allowed_special}

        if not special_tokens:
            return self._encode_normal(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens.keys()) + ")"
        parts = re.split(special_pattern, text)

        ids = []
        for part in parts:
            if part in special_tokens:
                ids.append(special_tokens[part])
            else:
                ids.extend(self._encode_normal(part))
        return ids

    def _encode_normal(self, text):
        chunks = self.compiled_pattern.findall(text)
        all_ids = []
        for chunk in chunks:
            all_ids.extend(self._encode_piece(chunk.encode("utf-8")))
        return all_ids

    def save(self, prefix):
        model = {
            "pattern": self.pattern,
            "encoder": {base64.b64encode(k).decode("ascii"): v for k, v in self.encoder.items()},
            "special_tokens": self.special_tokens
        }
        with open(f"{prefix}.json", "w") as f:
            json.dump(model, f)

    def load(self, prefix):
        with open(f"{prefix}.json", "r") as f:
            model = json.load(f)
        self.pattern = model["pattern"]
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = model["special_tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.encoder = {base64.b64decode(k): v for k, v in model["encoder"].items()}
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.encoder) + len(self.special_tokens)
