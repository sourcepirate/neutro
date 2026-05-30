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
from neutro.tokenizers import RegexTokenizer, get_gpt2_tokenizer

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

def train_wikitext_llm(tokenizer_type="bpe", vocab_size=2048, epochs=1):
    print("Loading WikiText-2...")
    text = load_wikitext2()
    
    if tokenizer_type == "gpt2":
        print("Loading GPT-2 Tokenizer...")
        tokenizer = get_gpt2_tokenizer()
    elif tokenizer_type == "bpe":
        print(f"Training custom BPE Tokenizer (vocab_size={vocab_size})...")
        tokenizer = RegexTokenizer()
        # Train on a portion of the text for speed
        tokenizer.train(text[:100000], vocab_size=vocab_size, verbose=True)
        print("BPE Training complete.")
    else:
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

    model.compile(optimizer=AdamW(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_accuracy'])
    model.summary()

    print("Starting training (Subset)...")
    # Small subset for demo
    x_sub = x[:1000]
    y_sub = y[:1000]
    
    # We no longer need one-hot encoding!
    
    model.fit(x_sub, y_sub, epochs=epochs, batch_size=32)

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
        indices = np.append(indices, next_idx)
    
    generated = tokenizer.decode(indices)
    print(f"Generated text: {generated}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Transformer LLM on WikiText-2")
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["char", "bpe", "gpt2"], help="Tokenizer type")
    parser.add_argument("--vocab_size", type=int, default=2048, help="Vocab size for BPE tokenizer")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()
    
    train_wikitext_llm(tokenizer_type=args.tokenizer, vocab_size=args.vocab_size, epochs=args.epochs)
