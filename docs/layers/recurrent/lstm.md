# Long Short-Term Memory (LSTM)

## What does this layer do?

LSTM (Long Short-Term Memory) is a recurrent layer designed to solve the **vanishing gradient problem** that plagues simple RNNs. It introduces a **cell state** — a "memory highway" that can carry information across many timesteps with minimal gradient decay — controlled by four learned gates.

### The math

At each timestep $t$, with input $x_t$, previous hidden state $h_{t-1}$, and previous cell state $C_{t-1}$:

$$f_t = \sigma(x_t W_f + h_{t-1} U_f + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(x_t W_i + h_{t-1} U_i + b_i) \quad \text{(input gate)}$$
$$\tilde{C}_t = \tanh(x_t W_C + h_{t-1} U_C + b_C) \quad \text{(candidate cell state)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(cell state update)}$$
$$o_t = \sigma(x_t W_o + h_{t-1} U_o + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(hidden state)}$$

Let's unpack every symbol:

- **$f_t$** — Forget gate. Values in `(0, 1)`. Controls how much of the *previous* cell state $C_{t-1}$ to keep. A value near 1 means "remember everything"; near 0 means "forget everything."
- **$i_t$** — Input gate. Values in `(0, 1)`. Controls how much of the *candidate* $\tilde{C}_t$ to add to the cell state.
- **$\tilde{C}_t$** — Candidate cell state. A proposed update to the cell state, computed like a simple RNN hidden state.
- **$C_t$** — Cell state. The "memory highway." Updated as: forget part of the old state, then add new information.
- **$o_t$** — Output gate. Values in `(0, 1)`. Controls how much of the cell state to expose as the hidden state $h_t$.
- **$h_t$** — Hidden state. The output at this timestep. A gated view of the cell state.

### Efficacy trick: one matrix to rule all gates

```python
self.params['W'] = init((self.features + self.units, 4 * self.units))
```

Instead of 8 separate weight matrices (W_f, W_i, W_C, W_o, U_f, U_i, U_C, U_o), LSTM concatenates the input and hidden state, then uses ONE matrix multiply with a `4 * units`-wide matrix:

$$
\text{concat}(x_t, h_{t-1}) \cdot W
$$

Then split the result into four equal chunks for i, f, c_tilde, o:

```
z → [  i  |  f  | c_tilde |  o  ]
     :units :2U  :3U        :4U
```

This is purely an optimization — one big matrix multiply is faster than eight small ones — but mathematically it's equivalent.

---

## Walking through the code

### `__init__` / `build`

```python
class LSTM(Layer):
    def __init__(self, units, return_sequences=False):
        super().__init__()
        self.units = units
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.features = input_shape[-1]
        init = get_initializer('glorot_uniform')
        self.params['W'] = init((self.features + self.units, 4 * self.units))
        self.params['b'] = get_initializer('zeros')((4 * self.units,))
        super().build(input_shape)
```

🔍 **`W` shape `(features + units, 4 * units)`** — This is the concatenated input-to-hidden AND hidden-to-hidden weight matrix.

📐 If `features = 32` and `units = 64`, then `W` is `(96, 256)`:

```
W = [W_input | W_hidden]   (first features rows from x, next units rows from h)
          ↑                      ↑
    shape (32, 256)         shape (64, 256)
```

Each of the four gates gets `units` columns: columns `0:64` → i-gate, `64:128` → f-gate, `128:192` → c_tilde, `192:256` → o-gate.

🔍 **`b` shape `(4 * units,)`** — One bias per gate. Split the same way as the columns.

Why no separate `U` matrix here? Because we concatenate `x_t` and `h_{t-1}` first, then multiply by the single `W`. The top `features` rows serve as the input weight, the bottom `units` rows serve as the recurrent weight.

### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    batch, timesteps, _ = inputs.shape
    self.h_states = np.zeros((batch, timesteps + 1, self.units))
    self.c_states = np.zeros((batch, timesteps + 1, self.units))
    self.gates = np.zeros((batch, timesteps, 4 * self.units))

    for t in range(timesteps):
        concat = np.concatenate([inputs[:, t, :], self.h_states[:, t, :]], axis=1)
        z = np.dot(concat, self.params['W']) + self.params['b']
        self.gates[:, t, :] = z
        i, f, c_tilde, o = self._sigmoid(z[:, :self.units]), \
                            self._sigmoid(z[:, self.units:2*self.units]), \
                            np.tanh(z[:, 2*self.units:3*self.units]), \
                            self._sigmoid(z[:, 3*self.units:])
        self.c_states[:, t+1, :] = f * self.c_states[:, t, :] + i * c_tilde
        self.h_states[:, t+1, :] = o * np.tanh(self.c_states[:, t+1, :])
    return self.h_states[:, 1:, :] if self.return_sequences else self.h_states[:, -1, :]
```

🔍 **Line 29**: `self.h_states = np.zeros((batch, timesteps + 1, self.units))` — Hidden states, same `T+1` pattern as SimpleRNN: index `0` is `h_0 = 0`.

🔍 **Line 30**: `self.c_states = np.zeros((batch, timesteps + 1, self.units))` — Cell states, also `T+1`. Index `0` is `C_0 = 0`. **Both** `h_states` and `c_states` are cached for backward.

🔍 **Line 31**: `self.gates = np.zeros((batch, timesteps, 4 * self.units))` — All four gate pre-activation values for every timestep. Cached for backward.

---

**Inside the loop (t = 0 to T-1):**

🔍 **Line 34**: `concat = np.concatenate([inputs[:, t, :], h_states[:, t, :]], axis=1)`

📐 `inputs[:, t, :]` is `(batch, features)`, `h_states[:, t, :]` is `(batch, units)`. Concatenated: `(batch, features + units)`.

🔍 **Line 35**: `z = np.dot(concat, W) + b`

📐 `(batch, features+units) @ (features+units, 4*units)` → `(batch, 4*units)`.

🔍 **Line 36**: `self.gates[:, t, :] = z` — Save the raw pre-activation for backward.

🔍 **Line 37**: Split `z` into four gates:

```python
i       = sigmoid(z[:, :units])              # input gate
f       = sigmoid(z[:, units:2*units])       # forget gate
c_tilde = tanh(z[:, 2*units:3*units])        # candidate
o       = sigmoid(z[:, 3*units:])            # output gate
```

Note: `i` and `f` use sigmoid (values 0 to 1), `c_tilde` uses tanh (values -1 to 1).

🔍 **Line 38**: `C_t = f * C_{t-1} + i * c_tilde`

📐 All four are shape `(batch, units)`. Element-wise operations.

This is the **cell state update**. The forget gate `f` decides how much of the old cell state to keep. The input gate `i` decides how much of the candidate to add. This is the "memory highway" — information can flow through unchanged when `f = 1` and `i = 0`.

🔍 **Line 39**: `h_t = o * tanh(C_t)`

📐 `(batch, units)`. The output gate controls how much of the cell state is exposed as the hidden state.

🔍 **Line 40**: Return all hidden states or just the last one, depending on `return_sequences`.

📐 Why cache ALL of `h_states`, `c_states`, and `gates`? Because `backward` needs:
- `h_states[:, t, :]` and `h_states[:, t+1, :]` for every `t`
- `c_states[:, t, :]` and `c_states[:, t+1, :]` for every `t`
- `gates[:, t, :]` (the pre-activation z) for every `t` to recompute the gate values

Without caching, backward would need to re-run the entire forward pass.

### `backward` (BPTT)

```python
def backward(self, grad_output):
    batch, timesteps, _ = self.inputs.shape
    d_W, d_b, d_inputs = np.zeros_like(self.params['W']), \
                         np.zeros_like(self.params['b']), \
                         np.zeros_like(self.inputs)
    dh_next, dc_next = np.zeros((batch, self.units)), \
                       np.zeros((batch, self.units))

    for t in range(timesteps - 1, -1, -1):
        dh = (grad_output[:, t, :] if self.return_sequences
              else (grad_output if t == timesteps - 1 else 0)) + dh_next
        z = self.gates[:, t, :]
        i, f, c_tilde, o = self._sigmoid(z[:, :self.units]), \
                            self._sigmoid(z[:, self.units:2*self.units]), \
                            np.tanh(z[:, 2*self.units:3*self.units]), \
                            self._sigmoid(z[:, 3*self.units:])
        tanh_c = np.tanh(self.c_states[:, t+1, :])
        do, dc = dh * tanh_c, dh * o * (1 - tanh_c**2) + dc_next
        df, di, dc_tilde = dc * self.c_states[:, t, :], dc * c_tilde, dc * i
        dz = np.concatenate([di * i * (1 - i),
                             df * f * (1 - f),
                             dc_tilde * (1 - c_tilde**2),
                             do * o * (1 - o)], axis=1)
        concat = np.concatenate([self.inputs[:, t, :],
                                 self.h_states[:, t, :]], axis=1)
        d_W += np.dot(concat.T, dz)
        d_b += np.sum(dz, axis=0)
        d_concat = np.dot(dz, self.params['W'].T)
        d_inputs[:, t, :], dh_next, dc_next = \
            d_concat[:, :self.features], d_concat[:, self.features:], f * dc
    self.grads['W'], self.grads['b'] = d_W, d_b
    return d_inputs
```

🧠 **"The backward loop goes in REVERSE order of the forward loop — that's the 'through time' part of BPTT"**

🔍 **Line 47**: `for t in range(timesteps - 1, -1, -1)` — From `T-1` down to `0`.

🔍 **Line 48**: `dh = grad_output + dh_next` — Same two-source gradient as SimpleRNN and GRU: from the layer above AND from the future timestep.

---

**Step 1: Compute `dc` — gradient w.r.t. the cell state**

$$h_t = o_t \cdot \tanh(C_t)$$

🔍 **Line 51-52**: Backprop through `h_t` to get the gradient for the output gate and the cell state:

```python
tanh_c = tanh(C_{t+1})
do = dh * tanh_c       # gradient w.r.t. output gate o
dc = dh * o * (1 - tanh_c**2) + dc_next
```

- `do = dh * tanh(C_t)` — By the product rule of `o * tanh(C_t)`.
- `dc` has **two sources**:
  1. `dh * o * (1 - tanh_c**2)` — The gradient through `tanh(C_t)` in the hidden state computation. The `(1 - tanh_c**2)` is the derivative of `tanh`.
  2. `dc_next` — The gradient from the *next* timestep's cell state (passed backward through `C_{t+1} = f * C_t + ...`).

📐 Both `(batch, units)`.

---

**Step 2: Backprop through the cell state update**

$$C_t = f \cdot C_{t-1} + i \cdot \tilde{C}_t$$

🔍 **Line 53**: Split `dc` into gradients for each gate:

```python
df = dc * C_{t-1}       # gradient w.r.t. forget gate f
di = dc * c_tilde       # gradient w.r.t. input gate i
dc_tilde = dc * i       # gradient w.r.t. candidate c_tilde
```

These are direct applications of the product rule. Each one is the gradient `dc` multiplied by the *other* operand in the sum.

🔍 Note how `dc_next` is computed at the end (line 59): `dc_next = f * dc`. This is the gradient of `C_t` w.r.t. `C_{t-1}` — the term `f` in `f * C_{t-1}` — which flows backward to the previous timestep.

---

**Step 3: Backprop through the activation functions**

Each gate has a different activation:

🔍 **Line 54**: `dz` is a concatenation of four gradient pieces, each multiplied by the derivative of their respective activation:

```python
dz = [
  di * i * (1 - i),              # sigmoid derivative for input gate
  df * f * (1 - f),              # sigmoid derivative for forget gate
  dc_tilde * (1 - c_tilde**2),   # tanh derivative for candidate
  do * o * (1 - o)               # sigmoid derivative for output gate
  ]
```

Concatenated: `(batch, 4*units)` — same shape as the original `z`.

---

**Step 4: Accumulate weight gradients**

🔍 **Line 55**: `concat = concat([inputs[:, t, :], h_states[:, t, :]])` — Same concatenation as forward.

📐 `(batch, features + units)`.

🔍 **Line 56**: `d_W += concat.T @ dz`

📐 `(features+units, batch) @ (batch, 4*units)` → `(features+units, 4*units)` = shape of `W`.

🔍 **Line 57**: `d_b += sum(dz, axis=0)` → `(4*units,)`.

---

**Step 5: Compute gradient w.r.t. inputs and previous hidden state**

🔍 **Line 58**: `d_concat = dz @ W.T`

📐 `(batch, 4*units) @ (4*units, features+units)` → `(batch, features+units)`.

🔍 **Line 59**: Split `d_concat` back into input gradient and hidden state gradient:

```python
d_inputs[:, t, :] = d_concat[:, :features]       # gradient for the previous layer
dh_next = d_concat[:, features:]                  # gradient for previous timestep's hidden state
dc_next = f * dc                                  # gradient for previous timestep's cell state
```

📐 `d_inputs[:, t, :]` is `(batch, features)`, `dh_next` is `(batch, units)`, `dc_next` is `(batch, units)`.

---

### Why LSTM solves vanishing gradients

The cell state update is a **linear highway**:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

When the forget gate `f` is close to 1 and the input gate `i` is close to 0, the cell state flows through **unchanged**:

$$C_t \approx C_{t-1}$$

The gradient of this path is:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

No repeated tanh or sigmoid squashing that shrinks gradients (like SimpleRNN's `(1 - h²)` factor). The gradient can flow backward through many timesteps without vanishing — that's the core insight of LSTMs.

In contrast, SimpleRNN's hidden state goes through `tanh` at every step:

$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$$

$$\frac{\partial h_t}{\partial h_{t-1}} = (1 - h_t^2) \cdot W_h$$

The `(1 - h_t^2)` factor is always ≤ 1, and multiplied across many timesteps, it drives gradients to zero.

---

## Try it yourself

```python
from neutro.layers.recurrent.lstm import LSTM
import numpy as np

# Create LSTM layer
lstm = LSTM(units=64, return_sequences=True)

# Input: batch of 4 sequences, each 10 timesteps, 32 features
x = np.random.randn(4, 10, 32)

# Forward pass
y = lstm(x)
print(f"Output shape: {y.shape}")            # (4, 10, 64)

# Backward pass
dL_dy = np.random.randn(4, 10, 64)
dL_dx = lstm.backward(dL_dy)
print(f"Input grad shape: {dL_dx.shape}")    # (4, 10, 32)
print(f"W grad shape: {lstm.grads['W'].shape}")  # (96, 256)
print(f"b grad shape: {lstm.grads['b'].shape}")  # (256,)

# With return_sequences=False
lstm_last = LSTM(units=64, return_sequences=False)
y_last = lstm_last(x)
print(f"Last-only output: {y_last.shape}")   # (4, 64)
```

## What to read next

- **`docs/layers/recurrent/recurrent.md`** — SimpleRNN and GRU: simpler recurrent architectures with walkthroughs of BPTT and gate mechanisms.
- **`docs/layers/attention/mha.md`** — Multi-Head Attention: the alternative to recurrence for sequence modeling (used in Transformers).
- **`docs/layers/core/dense.md`** — For a refresher on how `build`, `forward`, and `backward` work in the simplest layer.
