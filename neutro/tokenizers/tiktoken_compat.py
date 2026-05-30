import base64
import os
import urllib.request
from .bpe import RegexTokenizer

def load_tiktoken_bpe(tiktoken_bpe_file):
    """
    Loads a .tiktoken file and returns the mergeable ranks.
    """
    if tiktoken_bpe_file.startswith("http://") or tiktoken_bpe_file.startswith("https://"):
        # Download to a temporary location if it's a URL
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), "neutro_cache")
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(tiktoken_bpe_file)
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {tiktoken_bpe_file}...")
            with urllib.request.urlopen(tiktoken_bpe_file, timeout=30) as response, open(local_path, 'wb') as out_file:
                out_file.write(response.read())
        tiktoken_bpe_file = local_path

    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    
    ranks = {}
    for line in contents.splitlines():
        if not line:
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return ranks

class TiktokenCompatibleTokenizer(RegexTokenizer):
    """
    A tokenizer that can be initialized from tiktoken ranks.
    """
    def __init__(self, ranks, special_tokens, pattern=None):
        super().__init__(pattern=pattern)
        self.register_special_tokens(special_tokens)
        
        # Tiktoken already gives us exactly what we need: byte -> rank mapping
        # We ensure the base 256 bytes are present even if they weren't in ranks
        # although in standard tiktoken files they usually are.
        self.encoder = {bytes([i]): i for i in range(256)}
        self.encoder.update(ranks)
        self.decoder = {v: k for k, v in self.encoder.items()}


def get_gpt2_tokenizer():
    """
    Returns a TiktokenCompatibleTokenizer configured for GPT-2.
    """
    # GPT-2 ranks URL (r50k_base)
    GPT2_URL = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
    # GPT-2 pattern
    GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # GPT-2 special tokens
    special_tokens = {"<|endoftext|>": 50256}
    
    ranks = load_tiktoken_bpe(GPT2_URL)
    return TiktokenCompatibleTokenizer(ranks, special_tokens, pattern=GPT2_PATTERN)
