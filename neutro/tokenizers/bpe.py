import regex as re
import base64

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class BPETokenizer:
    """
    Minimal BPE Tokenizer implementation.
    Educational and "intentionally naive".
    """
    def __init__(self):
        # byte -> id mapping
        self.encoder = {bytes([i]): i for i in range(256)}
        self.decoder = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {} # str -> int
        self.inverse_special_tokens = {} # int -> str

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # text to bytes
        byte_chunks = [text.encode("utf-8")]
        
        # We need to maintain a list of token ids for each chunk
        ids_list = [list(chunk) for chunk in byte_chunks]

        for i in range(num_merges):
            stats = {}
            for ids in ids_list:
                for pair in zip(ids, ids[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
            if not stats:
                break
            
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Update chunks
            ids_list = [merge(ids, pair, idx) for ids in ids_list]
            
            # Update vocab
            new_token_bytes = self.decoder[pair[0]] + self.decoder[pair[1]]
            self.encoder[new_token_bytes] = idx
            self.decoder[idx] = new_token_bytes
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} had {stats[pair]} occurrences")

    def encode(self, text):
        return self._encode_piece(text.encode("utf-8"))

    def _encode_piece(self, piece_bytes):
        # Optimized BPE encoding using a list of token ids
        # Initially, each byte is a token
        ids = list(piece_bytes)
        
        while len(ids) >= 2:
            # Find the pair that would be merged first (lowest rank in encoder)
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.encoder.get(self.decoder[p[0]] + self.decoder[p[1]], float("inf")))
            
            # If the best pair is not in our merges, we are done
            pair_bytes = self.decoder[pair[0]] + self.decoder[pair[1]]
            if pair_bytes not in self.encoder:
                break
                
            idx = self.encoder[pair_bytes]
            ids = merge(ids, pair, idx)
            
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
    """
    BPE Tokenizer with Regex splitting (GPT style).
    """
    # GPT-4 split pattern
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmtre lve])| \?|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = pattern if pattern else self.GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # split text into chunks
        chunks = self.compiled_pattern.findall(text)
        # convert chunks to byte ids
        ids_list = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            stats = {}
            for ids in ids_list:
                # get stats within each chunk
                for pair in zip(ids, ids[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
            
            if not stats:
                break
            
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # merge in all chunks
            ids_list = [merge(ids, pair, idx) for ids in ids_list]
            
            new_token_bytes = self.decoder[pair[0]] + self.decoder[pair[1]]
            self.encoder[new_token_bytes] = idx
            self.decoder[idx] = new_token_bytes
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} had {stats[pair]} occurrences")

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
        import json
        model = {
            "pattern": self.pattern,
            "encoder": {base64.b64encode(k).decode("ascii"): v for k, v in self.encoder.items()},
            "special_tokens": self.special_tokens
        }
        with open(f"{prefix}.json", "w") as f:
            json.dump(model, f)
            
    def load(self, prefix):
        import json
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

