import pytest
import numpy as np
from neutro.tokenizers import BPETokenizer, RegexTokenizer

def test_bpe_tokenizer_basic():
    tokenizer = BPETokenizer()
    text = "hello world hello world"
    # Small vocab to force merges
    tokenizer.train(text, vocab_size=260) 
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    assert decoded == text
    assert len(encoded) < len(text.encode("utf-8"))
    assert max(encoded) < 260

def test_regex_tokenizer():
    tokenizer = RegexTokenizer()
    text = "Hello, world! 123"
    tokenizer.train(text, vocab_size=270)
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    assert decoded == text
    assert max(encoded) < 270

def test_regex_tokenizer_special_tokens():
    tokenizer = RegexTokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 1000})
    
    text = "Hello<|endoftext|>World"
    # training on some text
    tokenizer.train("Hello World", vocab_size=260)
    
    # encoded with special tokens allowed
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    assert 1000 in encoded
    
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_save_load():
    import os
    prefix = "test_tokenizer"
    tokenizer = RegexTokenizer()
    tokenizer.train("simple text for training", vocab_size=260)
    tokenizer.register_special_tokens({"[SEP]": 1001})
    
    tokenizer.save(prefix)
    
    new_tokenizer = RegexTokenizer()
    new_tokenizer.load(prefix)
    
    assert new_tokenizer.pattern == tokenizer.pattern
    assert new_tokenizer.special_tokens == tokenizer.special_tokens
    assert new_tokenizer.encoder == tokenizer.encoder
    assert new_tokenizer.decoder == tokenizer.decoder
    
    os.remove(f"{prefix}.json")

def test_tokenization_consistency():
    tokenizer = RegexTokenizer()
    # Train on a mix of text
    train_text = "The quick brown fox jumps over the lazy dog. 1234567890!@#$%^&*()"
    tokenizer.train(train_text, vocab_size=300)
    
    test_texts = [
        "The quick brown fox",
        "Hello World!",
        "123 456",
        "Complex symbols: @#$%^&*()",
        "Newline\nand\ttabs."
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Failed for text: {text}"
