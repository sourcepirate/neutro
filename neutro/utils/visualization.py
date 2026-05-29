import numpy as np

def plot_attention_weights(attention_weights, tokens=None, title="Attention Weights"):
    """
    Plots attention weights as a heatmap in the terminal using ASCII characters.
    Since we don't use matplotlib to keep it 'naive' and terminal-friendly.
    
    Args:
        attention_weights: (seq_len, seq_len) array.
        tokens: List of tokens (strings) for axis labels.
    """
    seq_len = attention_weights.shape[0]
    if tokens is None:
        tokens = [str(i) for i in range(seq_len)]
    
    # ASCII intensity scale
    chars = " .:-=+*#%@"
    
    print(f"\n--- {title} ---")
    
    # Print header
    header = "    " + "".join([f"{t[:3]:>4}" for t in tokens])
    print(header)
    
    for i in range(seq_len):
        row = f"{tokens[i][:3]:<4}"
        for j in range(seq_len):
            val = attention_weights[i, j]
            char_idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
            row += f"  {chars[char_idx]} "
        print(row)
    print("-" * len(header))

def print_text_with_probs(tokens, probs, threshold=0.1):
    """
    Prints text where tokens with high probability are highlighted.
    """
    for token, prob in zip(tokens, probs):
        if prob > threshold:
            print(f"\033[92m{token}\033[0m", end=" ")
        else:
            print(token, end=" ")
    print()
