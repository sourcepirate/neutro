import pytest
import base64
import os
import tempfile
from neutro.tokenizers.tiktoken_compat import TiktokenCompatibleTokenizer, load_tiktoken_bpe

def test_load_tiktoken_bpe():
    # Create a dummy .tiktoken file
    # Format: base64(token) rank
    merges = {
        b"he": 256,
        b"ll": 257,
        b"hello": 258
    }
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        for token, rank in merges.items():
            f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")
        temp_path = f.name

    try:
        ranks = load_tiktoken_bpe(temp_path)
        assert len(ranks) == 3
        assert ranks[b"hello"] == 258
    finally:
        os.remove(temp_path)

def test_tiktoken_compatible_tokenizer():
    # Simple ranks for testing
    # 0-255 are implicit
    ranks = {
        b"he": 256,
        b"ll": 257,
        b"hell": 258,
        b"hello": 259
    }
    special_tokens = {"<|endoftext|>": 1000}
    
    tokenizer = TiktokenCompatibleTokenizer(ranks, special_tokens)
    
    assert tokenizer.vocab_size == 256 + 4 + 1
    assert tokenizer.special_tokens["<|endoftext|>"] == 1000
    
    text = "hello"
    encoded = tokenizer.encode(text)
    # Reconstructed vocab should allow encoding 'hello'
    assert tokenizer.decode(encoded) == text
    
    # Test with special tokens
    text_with_special = "hello<|endoftext|>"
    encoded_special = tokenizer.encode(text_with_special, allowed_special="all")
    assert 1000 in encoded_special
    assert tokenizer.decode(encoded_special) == text_with_special


from unittest.mock import patch, MagicMock


@patch('urllib.request.urlopen')
@patch('os.path.exists')
@patch('os.makedirs')
@patch('tempfile.gettempdir')
@patch('builtins.open', new_callable=MagicMock)
def test_load_tiktoken_bpe_url(mock_file_open, mock_gettempdir, mock_makedirs, mock_exists, mock_urlopen):
    import base64
    
    mock_gettempdir.return_value = "/tmp"
    mock_exists.return_value = False
    
    content = base64.b64encode(b"hello") + b" 258\n" + base64.b64encode(b"world") + b" 259\n"
    
    mock_response = MagicMock()
    mock_response.read.return_value = content
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = content
    mock_file_open.return_value = mock_file
    
    from neutro.tokenizers.tiktoken_compat import load_tiktoken_bpe
    ranks = load_tiktoken_bpe("https://example.com/test.tiktoken")
    assert len(ranks) == 2
    assert ranks[b"hello"] == 258
    assert ranks[b"world"] == 259


@patch('neutro.tokenizers.tiktoken_compat.load_tiktoken_bpe')
def test_get_gpt2_tokenizer(mock_load_bpe):
    mock_load_bpe.return_value = {}
    
    from neutro.tokenizers.tiktoken_compat import get_gpt2_tokenizer
    tokenizer = get_gpt2_tokenizer()
    assert tokenizer is not None
    assert tokenizer.special_tokens["<|endoftext|>"] == 50256
    mock_load_bpe.assert_called_once()
