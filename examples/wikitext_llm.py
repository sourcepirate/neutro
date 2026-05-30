import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models import Sequential
from neutro.layers import Embedding, TransformerBlock, Dense, Softmax, Flatten
from neutro.utils.data_utils import load_wikitext2
from neutro.optimizers import AdamW
from neutro.layers.attention.base_attention import BaseAttention
from neutro.data import DataLoader

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return np.array([self.char_to_idx[ch] for ch in text if ch in self.char_to_idx])

    def decode(self, indices):
        return "".join([self.idx_to_char[i] for i in indices])

def prepare_data(text, tokenizer, seq_len=64, step=3):
    encoded = tokenizer.encode(text)
    # Subset for demo
    encoded = encoded[:100000]
    
    x, y = [], []
    for i in range(0, len(encoded) - seq_len, step):
        x.append(encoded[i:i + seq_len])
        y.append(encoded[i + 1:i + seq_len + 1])
    
    return np.array(x), np.array(y)

def train_wikitext_llm():
    print("Loading WikiText-2...")
    text = load_wikitext2()
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    seq_len = 32
    
    print(f"Vocab size: {vocab_size}")
    
    print("Preparing data...")
    x, y = prepare_data(text, tokenizer, seq_len=seq_len)
    
    # One-hot encode targets for categorical_crossentropy
    # To save memory in demo, we could use a custom loss or smaller subset
    # But let's follow the API. (batch, seq, vocab)
    
    # Note: Our current categorical_crossentropy expects (batch, classes)
    # For sequence tasks, we might need to flatten or update the loss.
    # Let's check the loss implementation.
    
    print("Building Transformer LLM...")
    model = Sequential([
        Embedding(vocab_size, 128, input_shape=(seq_len,)),
        TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, causal=True, use_flash=True),
        TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, causal=True, use_flash=True),
        Dense(vocab_size),
        Softmax()
    ])

    model.compile(optimizer=AdamW(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Custom training loop or adjust Y for Sequential.fit
    # Sequential.fit currently expects Y to match output shape.
    # Output is (batch, seq_len, vocab_size)
    
    # Let's adjust the fit method or targets. 
    # Current CategoricalCrossentropy probably handles (batch, seq, vocab) if flattened.
    
    print("Starting training (Subset)...")
    # Small subset for demo
    x_sub = x[:1000]
    y_sub = y[:1000]
    
    # One-hot encode Y: (1000, seq_len, vocab_size)
    y_onehot = np.zeros((len(y_sub), seq_len, vocab_size))
    for i in range(len(y_sub)):
        for j in range(seq_len):
            y_onehot[i, j, y_sub[i, j]] = 1.0

    # We need to flatten targets and predictions for standard categorical_crossentropy
    # Or update the loss to handle sequences.
    # Actually, Sequential.fit and CategoricalCrossentropy in this lib
    # might need a Flatten layer before Dense if we want (batch, vocab)
    # but for LLM we want (batch, seq, vocab).
    
    # Let's check neutro/losses/categorical_crossentropy.py
    model.fit(x_sub, y_onehot, epochs=5, batch_size=32)

    print("\nGenerating text...")
    start_text = "The "
    generated = start_text
    indices = tokenizer.encode(start_text)
    
    for _ in range(50):
        # Pad or slice to seq_len
        curr_x = indices[-seq_len:]
        if len(curr_x) < seq_len:
            curr_x = np.pad(curr_x, (seq_len - len(curr_x), 0))
        
        preds = model.predict(curr_x.reshape(1, -1)) # (1, seq, vocab)
        next_idx = np.argmax(preds[0, -1, :])
        generated += tokenizer.idx_to_char[next_idx]
        indices = np.append(indices, next_idx)
    
    print(f"Generated text: {generated}")

if __name__ == "__main__":
    train_wikitext_llm()
