import pytest
import numpy as np
from neutro.models.language.llama import LlamaTiny
from neutro.models.language.deepseek import DeepSeekV2Tiny
from neutro.models.base_model import Sequential
from neutro.layers.core.dense import Dense

def test_summary_llama():
    model = LlamaTiny(vocab_size=1000, seq_len=32, dim=64, n_layers=2, n_heads=4)
    # This should print without error
    model.summary()
    
    # Check if total params is > 0
    total = 0
    for layer in model.layers:
        total += layer.count_params()
    
    assert total > 0

def test_summary_deepseek():
    model = DeepSeekV2Tiny(vocab_size=1000, seq_len=32, dim=64, n_layers=1, n_heads=4)
    model.summary()
    
    total = 0
    for layer in model.layers:
        total += layer.count_params()
        
    assert total > 0

def test_nested_training_discovery():
    model = LlamaTiny(vocab_size=1000, seq_len=32, dim=64, n_layers=1, n_heads=4)
    all_layers = model._get_all_layers()
    
    # LlamaBlock has attention, ffn, norms
    # LlamaMLP has w1, w2, w3
    # Check if we found w1 inside the block
    found_w1 = False
    for layer in all_layers:
        if isinstance(layer, Dense) and hasattr(layer, 'units'):
            # LlamaMLP uses hidden_dim which is dim*4 usually
            if layer.units == 64 * 4:
                found_w1 = True
                break
    
    assert found_w1, "Nested layers not discovered by _get_all_layers"

if __name__ == "__main__":
    test_summary_llama()
    test_summary_deepseek()
    test_nested_training_discovery()
