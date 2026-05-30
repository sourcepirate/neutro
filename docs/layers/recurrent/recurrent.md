# Recurrent Layers: SimpleRNN, LSTM, GRU

## Theory

Recurrent Neural Networks process sequences by maintaining a hidden state that is updated at each time step. The key challenge is the **vanishing gradient problem** — gradients diminish exponentially over long sequences.

### SimpleRNN — `neutro/layers/recurrent/simple_rnn.py`

$$h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b)$$

Simple RNN suffers from vanishing gradients and cannot capture long-range dependencies.

### LSTM — `neutro/layers/recurrent/lstm.py`

Long Short-Term Memory introduces a gating mechanism with a cell state:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(candidate)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(cell update)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(hidden state)}$$

The cell state $C_t$ can carry information over long distances with minimal gradient decay.

### GRU — `neutro/layers/recurrent/gru.py`

Gated Recurrent Unit simplifies LSTM by merging the cell state and hidden state:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

GRU has fewer parameters than LSTM and often performs comparably.

## Implementation Guide

All recurrent layers are in `neutro/layers/recurrent/`. They share a common pattern:

```python
def forward(self, inputs, training=False):
    batch_size, seq_len, input_dim = inputs.shape
    # Initialize hidden state
    h = np.zeros((batch_size, self.units))
    self.h_states = []
    for t in range(seq_len):
        x_t = inputs[:, t, :]
        h = self._step(x_t, h)  # One RNN step
        self.h_states.append(h)
    return np.stack(self.h_states, axis=1)
```

The backward pass (Backpropagation Through Time, BPTT) reverses the loop:

```python
def backward(self, grad_output):
    for t in reversed(range(self.seq_len)):
        grad_h = grad_output[:, t, :] + grad_h_next
        # Backprop through one step
        ...
        grad_h_next = grad_from_h
    return grad_x
```

Weight concatenation optimization (LSTM, line 53): weights for all four gates are stored as a single matrix to optimize the dot product: `W = np.concatenate([W_f, W_i, W_C, W_o])`.

## Usage Example

```python
from neutro.layers import LSTM, GRU

lstm = LSTM(units=128, return_sequences=True)
x = np.random.randn(4, 32, 64)  # (batch, seq, features)
y = lstm(x)                      # (batch, seq, 128)

gru = GRU(units=64, return_sequences=False)
z = gru(x)                       # (batch, 64)
```

## References

- Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*.
- Chung, J., et al. (2014). **Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling**. [arXiv:1412.3555](https://arxiv.org/abs/1412.3555)
