import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models import Sequential
from neutro.layers import Embedding, GRU, Dense, Softmax
from neutro.optimizers import Adam
from neutro.utils.data_utils import load_imdb, get_imdb_word_index
from neutro.preprocessing.sequence import pad_sequences

def decode_review(indices, reverse_word_index):
    # IMDB indices are offset by 3
    # 0 is padding, 1 is start, 2 is OOV
    return " ".join([reverse_word_index.get(i - 3, "?") for i in indices if i > 2])

def train_gru_sentiment():
    print("Loading IMDB dataset...")
    # This might take a moment to download (~80MB)
    (x_train, y_train), (x_test, y_test) = load_imdb()
    word_index = get_imdb_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    # Shuffle before taking subsets
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Keep original test data for printing human readable reviews later
    test_indices = np.arange(len(x_test))
    np.random.shuffle(test_indices)
    x_test_raw = x_test[test_indices]
    y_test_raw = y_test[test_indices]
    
    vocab_size = 10000
    maxlen = 100
    
    print(f"Preprocessing data (maxlen={maxlen})...")
    # IMDB indices are already provided, we just need to pad/truncate
    # and filter by vocab size if necessary (the npz usually has up to 88k)
    # Let's clip indices to vocab_size
    def preprocess_set(data):
        return [np.array([i if i < vocab_size else 2 for i in s]) for s in data]

    x_train_processed = pad_sequences(preprocess_set(x_train_shuffled), maxlen=maxlen)
    x_test_processed = pad_sequences(preprocess_set(x_test_raw), maxlen=maxlen)
    
    # One-hot encode targets
    y_train_cat = np.eye(2)[y_train_shuffled]
    y_test_cat = np.eye(2)[y_test_raw]
    
    # Subset for faster demo in NumPy
    n_train = 1000
    n_test = 200
    x_train_subset, y_train_subset = x_train_processed[:n_train], y_train_cat[:n_train]
    x_test_subset, y_test_subset = x_test_processed[:n_test], y_test_cat[:n_test]

    print("Building GRU Model...")
    model = Sequential([
        Embedding(vocab_size, 64, input_shape=(maxlen,)),
        GRU(64, return_sequences=False),
        Dense(2),
        Softmax()
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting training (Subset of 1000 samples)...")
    model.fit(x_train_subset, y_train_subset, epochs=5, batch_size=32, validation_data=(x_test_subset, y_test_subset))
    
    print("Evaluating...")
    results = model.evaluate(x_test_subset, y_test_subset)
    print(f"Test Results: {results}")

    # Print some reviews and their predictions
    print("\n" + "="*50)
    print("DEMO: PREDICTIONS ON TEST SAMPLES")
    print("="*50)
    
    for i in range(5):
        raw_review_indices = x_test_raw[i]
        actual_sentiment = "Positive" if y_test_raw[i] == 1 else "Negative"
        
        # Preprocess single sample
        proc_review = np.array([i if i < vocab_size else 2 for i in raw_review_indices])
        proc_review = pad_sequences([proc_review], maxlen=maxlen)
        
        # Predict
        pred_probs = model.predict(proc_review)
        pred_sentiment = "Positive" if np.argmax(pred_probs) == 1 else "Negative"
        confidence = np.max(pred_probs) * 100
        
        review_text = decode_review(raw_review_indices, reverse_word_index)
        # Truncate text for display
        display_text = (review_text[:150] + '...') if len(review_text) > 150 else review_text
        
        print(f"Review: \"{display_text}\"")
        print(f"Actual: {actual_sentiment} | Predicted: {pred_sentiment} ({confidence:.1f}%)")
        print("-" * 50)

    # Test with a manual sequence
    print("\nPredicting on manual sample...")
    # 0 is usually padding, 1 is start, 2 is OOV
    # "this movie was great" (indices would depend on word_index, but let's just use some)
    sample_seq = np.array([[1, 14, 22, 16, 84, 0, 0, 0, 0, 0]]) 
    sample_seq = pad_sequences(sample_seq, maxlen=maxlen)
    pred = model.predict(sample_seq)
    sentiment = "Positive" if np.argmax(pred) == 1 else "Negative"
    print(f"Review prediction: {sentiment} (Prob: {np.max(pred):.4f})")

if __name__ == "__main__":
    train_gru_sentiment()
