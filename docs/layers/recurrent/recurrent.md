# SimpleRNN and GRU

## SimpleRNN

### What does this layer do?

An RNN processes sequences one step at a time, maintaining a hidden state that carries information forward. At each timestep, it combines the current input with the previous hidden state through a learned transformation.

### The math

$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$$

Let's unpack every symbol:

- **$x_t$** — The input at timestep `t`. Shape `(batch, features)`. Each timestep's slice of the input sequence.
- **$W_x$** — Input weight matrix. Shape `(features, units)`. Controls how the current input influences the new hidden state.
- **$h_{t-1}$** — The hidden state from the *previous* timestep. Shape `(batch, units)`. This is the "memory" that carries information across timesteps.
- **$W_h$** — Recurrent weight matrix. Shape `(units, units)`. Controls how the previous hidden state influences the new one.
- **$b$** — Bias vector. Shape `(units,)`.
- **$\tanh$** — Hyperbolic tangent activation, squashing values into `(-1, 1)`.
- **$h_t$** — The new hidden state. Also the output at this timestep (if `return_sequences=True`).

### Walking through the code

#### `__init__` / `build`

```python
def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.return_sequences = return_sequences
    self.activation_name = activation

def build(self, input_shape):
    self.features = input_shape[-1]
    init = get_initializer('glorot_uniform')
    self.params['Wx'] = init((self.features, self.units))
    self.params['Wh'] = init((self.units, self.units))
    self.params['b'] = get_initializer('zeros')((self.units,))
    super().build(input_shape)
```

🔍 **`features = input_shape[-1]`** — We grab the last dimension of the input. If input is `(batch, timesteps, features)`, this is the feature dimension. The first two (batch and timesteps) are handled dynamically.

🔍 **`Wx` shape `(features, units)`** — Maps `features` input dimensions → `units` hidden dimensions. If input is 128-dimensional and we want 64 hidden units, this is `(128, 64)`.

🔍 **`Wh` shape `(units, units)`** — Maps hidden state → hidden state. A square matrix: `(64, 64)`. This is what makes RNNs "recurrent" — the same hidden-to-hidden weight is applied at every timestep.

🔍 **`b` shape `(units,)`** — One bias per hidden unit. Broadcast across batch.

#### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    batch, timesteps, _ = inputs.shape
    self.h_states = np.zeros((batch, timesteps + 1, self.units))

    for t in range(timesteps):
        z = np.dot(inputs[:, t, :], self.params['Wx']) + \
            np.dot(self.h_states[:, t, :], self.params['Wh']) + self.params['b']
        if self.activation_name == 'tanh':
            self.h_states[:, t+1, :] = np.tanh(z)
        else:
            self.h_states[:, t+1, :] = z
    if self.return_sequences:
        return self.h_states[:, 1:, :]
    return self.h_states[:, -1, :]
```

🔍 **Line 27**: `self.inputs = inputs` — Cached for backward. `backward` doesn't receive the original inputs — it only gets `grad_output` — so we must save `inputs` now.

🔍 **Line 28**: `self.h_states = np.zeros((batch, timesteps + 1, self.units))` — We allocate space for **all** hidden states, one per timestep *plus one extra* for the initial `h_0 = 0`. Shape: `(batch, T+1, units)`.

Why `T+1`? So `h_states[:, t, :]` and `h_states[:, t+1, :]` both index validly in the loop, where `t` goes `0` to `T-1`. Entry `0` is `h_0` (all zeros), entry `1` is `h_1`, ..., entry `T` is `h_T`.

🔍 **Line 31**: The core computation:

📐 `inputs[:, t, :]` is `(batch, features)` @ `Wx` `(features, units)` → `(batch, units)`

📐 `h_states[:, t, :]` is `(batch, units)` @ `Wh` `(units, units)` → `(batch, units)`

📐 Adding them plus bias (broadcast) gives `z` of shape `(batch, units)`.

🔍 **Line 32-35**: `np.tanh(z)` applies the squashing non-linearity. The result is stored at position `t+1` in `h_states`, making it `h_t`.

🔍 **Lines 36-38**: If `return_sequences=True`, return all hidden states `h_1` through `h_T` — shape `(batch, T, units)`. If `False`, return only the last hidden state `h_T` — shape `(batch, units)`.

📐 **Output shapes**:
- `return_sequences=True`: `(batch, timesteps, units)`
- `return_sequences=False`: `(batch, units)`

🔍 Why do we cache ALL hidden states (not just the last one)? Because `backward` needs `h_t` and `h_{t+1}` for every `t` to compute gradients. Without caching, we'd have to re-run the forward loop.

#### `backward` (BPTT — Backpropagation Through Time)

```python
def backward(self, grad_output):
    batch, timesteps, _ = self.inputs.shape
    d_Wx, d_Wh, d_b = np.zeros_like(self.params['Wx']), \
                      np.zeros_like(self.params['Wh']), \
                      np.zeros_like(self.params['b'])
    d_inputs = np.zeros_like(self.inputs)
    dh_next = np.zeros((batch, self.units))

    for t in range(timesteps - 1, -1, -1):
        dh = (grad_output[:, t, :] if self.return_sequences \
              else (grad_output if t == timesteps - 1 else 0)) + dh_next
        dz = dh * (1 - self.h_states[:, t+1, :]**2)
        d_Wx += np.dot(self.inputs[:, t, :].T, dz)
        d_Wh += np.dot(self.h_states[:, t, :].T, dz)
        d_b += np.sum(dz, axis=0)
        d_inputs[:, t, :] = np.dot(dz, self.params['Wx'].T)
        dh_next = np.dot(dz, self.params['Wh'].T)

    self.grads['Wx'], self.grads['Wh'], self.grads['b'] = d_Wx, d_Wh, d_b
    return d_inputs
```

🧠 **"The backward loop goes in REVERSE order of the forward loop — that's the 'through time' part of BPTT"**

🔍 **Line 46**: `for t in range(timesteps - 1, -1, -1)` — Loop from `T-1` down to `0`. The forward loop went `0, 1, 2, ..., T-1`. The backward loop goes `T-1, ..., 2, 1, 0`.

🔍 **Line 47**: `dh = grad_output + dh_next` — The gradient arriving at this timestep has **two sources**:

1. **From above**: the upstream gradient `grad_output` from the loss (or next layer). If `return_sequences=True`, each timestep `t` gets its own slice `grad_output[:, t, :]`. If `return_sequences=False`, only the last timestep gets the gradient; all others get a zero contribution from this source.

2. **From the future**: `dh_next` — the gradient flowing *back* from timestep `t+1`. This is computed on the **previous iteration** of the backward loop (which was timestep `t+1`, since we're going in reverse).

This two-source pattern is the essence of BPTT — future timesteps send gradient information backward through `Wh`.

🔍 **Line 48**: `dz = dh * (1 - h_{t+1}^2)` — This is the derivative of $\tanh$: $\frac{d}{dz}\tanh(z) = 1 - \tanh(z)^2 = 1 - h_{t+1}^2$.

We multiply `dh` by this derivative (element-wise) to backpropagate through the tanh activation. This is the chain rule: $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z}$.

🔍 **Lines 49-51**: Accumulate gradients for each parameter:

- `d_Wx += x_t^T @ dz` — Shape: `(features, batch) @ (batch, units)` → `(features, units)` = `Wx` shape.
- `d_Wh += h_t^T @ dz` — Shape: `(units, batch) @ (batch, units)` → `(units, units)` = `Wh` shape.
- `d_b += sum(dz, axis=0)` — Sum over batch → `(units,)` = `b` shape.

These are **accumulated** (with `+=`) across all timesteps because the same `Wx`, `Wh`, and `b` are used at every timestep — the total gradient is the sum of the gradients from each timestep.

🔍 **Line 52**: `d_inputs[:, t, :] = dz @ Wx.T` — The gradient w.r.t. the input at this timestep. Shape: `(batch, units) @ (units, features)` → `(batch, features)`.

🔍 **Line 53**: `dh_next = dz @ Wh.T` — The gradient to pass to the previous timestep. Shape: `(batch, units) @ (units, units)` → `(batch, units)`. This becomes `dh_next` in the `t-1` iteration.

📐 Gradient shapes flowing backward:

```
t = T-1:
  grad_output  →  dh (batch, units)  →  dz (batch, units)
                                         → d_Wx (features, units)
                                         → d_Wh (units, units)
                                         → d_b (units,)
                                         → d_inputs[:, T-1, :] (batch, features)
                                         → dh_next (batch, units)
                                                         ↓
t = T-2:                                    dh_next passed to previous step
  grad_output  +  dh_next  →  dh  →  dz  → ... same pattern
```

---

## GRU

### What does this layer do?

A Gated Recurrent Unit (GRU) is a simplified version of LSTM that merges the cell state and hidden state. It uses two gates — an **update gate** (how much to keep vs. replace) and a **reset gate** (how much to forget the past) — to control information flow.

### The math

$$z_t = \sigma(x_t W_z + h_{t-1} U_z) \quad \text{(update gate)}$$

$$r_t = \sigma(x_t W_r + h_{t-1} U_r) \quad \text{(reset gate)}$$

$$\tilde{h}_t = \tanh(x_t W_h + (r_t \odot h_{t-1}) U_h) \quad \text{(candidate hidden state)}$$

$$h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1} \quad \text{(final hidden state)}$$

Let's unpack every symbol:

- **$z_t$** — Update gate. Values in `(0, 1)`. A value near 1 means "keep the old state"; near 0 means "replace with the candidate." Note: the naming convention is sometimes flipped — here $z$ controls how much of the *old* state to keep, so $(1-z)$ controls how much of the *new* candidate to take.
- **$r_t$** — Reset gate. Values in `(0, 1)`. Controls how much of the past hidden state to forget when computing the candidate.
- **$\tilde{h}_t$** — Candidate hidden state. Like a regular RNN's hidden state, but modulated by the reset gate.
- **$W_z, W_r, W_h$** — Input weight matrices. Each has shape `(features, units)`. They are stacked into one big matrix `W` of shape `(features, 3 * units)` for efficiency.
- **$U_z, U_r, U_h$** — Recurrent weight matrices. Each has shape `(units, units)`. Stacked into `U` of shape `(units, 3 * units)`.
- **$h_t$** — Final hidden state. An interpolation between the old state and the candidate, controlled by `z`.

### Walking through the code

#### `__init__` / `build`

```python
def __init__(self, units, return_sequences=False, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.return_sequences = return_sequences

def build(self, input_shape):
    self.features = input_shape[-1]
    init = get_initializer('glorot_uniform')
    self.params['W'] = init((self.features, 3 * self.units))
    self.params['U'] = init((self.units, 3 * self.units))
    self.params['b'] = get_initializer('zeros')((3 * self.units,))
    super().build(input_shape)
```

🔍 **Weight organization**: Instead of 6 separate matrices (Wz, Wr, Wh, Uz, Ur, Uh), the GRU concatenates them:

```
W = [Wz | Wr | Wh]    shape: (features, 3*units)
U = [Uz | Ur | Uh]    shape: (units, 3*units)
b = [bz | br | bh]    shape: (3*units,)
```

The first `2*units` columns are for the update and reset gates (z, r). The last `units` columns are for the candidate hidden state (h_tilde).

📐 **Why concatenate?** A single matrix multiply `x @ W` is faster than three separate ones. We split the result afterward.

#### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    batch, timesteps, _ = inputs.shape
    self.h_states = np.zeros((batch, timesteps + 1, self.units))

    self.z_gates = np.zeros((batch, timesteps, self.units))
    self.r_gates = np.zeros((batch, timesteps, self.units))
    self.h_tilde = np.zeros((batch, timesteps, self.units))
    self.x_W = np.dot(inputs, self.params['W'])  # (batch, timesteps, 3*units)

    for t in range(timesteps):
        x_W_t = self.x_W[:, t, :] + self.params['b']
        h_prev = self.h_states[:, t, :]

        z_r_hidden = np.dot(h_prev, self.params['U'][:, :2*self.units])
        z_r_logits = x_W_t[:, :2*self.units] + z_r_hidden
        z_r = self._sigmoid(z_r_logits)

        z = z_r[:, :self.units]
        r = z_r[:, self.units:]

        self.z_gates[:, t, :] = z
        self.r_gates[:, t, :] = r

        h_tilde_hidden = np.dot(r * h_prev, self.params['U'][:, 2*self.units:])
        h_tilde_logits = x_W_t[:, 2*self.units:] + h_tilde_hidden
        h_tilde = np.tanh(h_tilde_logits)
        self.h_tilde[:, t, :] = h_tilde

        self.h_states[:, t+1, :] = (1 - z) * h_tilde + z * h_prev

    if self.return_sequences:
        return self.h_states[:, 1:, :]
    return self.h_states[:, -1, :]
```

🔍 **Lines 47-49**: Pre-allocate storage for **all** intermediate values — `z_gates`, `r_gates`, `h_tilde` across all timesteps. These are cached for the backward pass.

🔍 **Line 50**: `self.x_W = np.dot(inputs, self.params['W'])` — Compute the input-to-hidden projection for **all timesteps at once**.

📐 `inputs` is `(batch, timesteps, features)` @ `W` is `(features, 3*units)` → `(batch, timesteps, 3*units)`.

This is an optimization: one big matrix multiply instead of `T` small ones. At timestep `t`, we slice `self.x_W[:, t, :]`.

🔍 **Lines 57-62**: Compute the update gate `z` and reset gate `r`:

1. `z_r_hidden = h_prev @ U[:, :2*units]` — The recurrent part of the gates.
2. `z_r_logits = x_W_t[:, :2*units] + z_r_hidden` — Add the input part (from the pre-computed `x_W`).
3. `z_r = sigmoid(z_r_logits)` — Both gates share one sigmoid computation.
4. Split: `z = z_r[:, :units]`, `r = z_r[:, units:]`.

🔍 **Lines 64-71**: Compute the candidate hidden state:

1. `h_tilde_hidden = (r * h_prev) @ U[:, 2*units:]` — The reset gate element-wise multiplies the previous hidden state, zeroing out some dimensions.
2. `h_tilde_logits = x_W_t[:, 2*units:] + h_tilde_hidden` — Add the input part.
3. `h_tilde = tanh(h_tilde_logits)` — Squash to `(-1, 1)`.

🔍 **Line 74**: `h_t = (1 - z) * h_tilde + z * h_prev` — The final hidden state is an **interpolation** between the old state and the candidate. When `z` is close to 1, we mostly keep the old state. When `z` is close to 0, we mostly take the new candidate.

📐 **Why cache `z_gates`, `r_gates`, `h_tilde`, and `x_W`?** Because `backward` needs every gate value at every timestep, plus the input projection. Without caching them all, backward would have to recompute the entire forward pass.

#### `backward` (BPTT)

```python
def backward(self, grad_output):
    batch, timesteps, _ = self.inputs.shape
    d_W = np.zeros_like(self.params['W'])
    d_U = np.zeros_like(self.params['U'])
    d_b = np.zeros_like(self.params['b'])
    d_inputs = np.zeros_like(self.inputs)
    dh_next = np.zeros((batch, self.units))

    for t in range(timesteps - 1, -1, -1):
        dh = (grad_output[:, t, :] if self.return_sequences
              else (grad_output if t == timesteps - 1 else 0)) + dh_next

        z = self.z_gates[:, t, :]
        r = self.r_gates[:, t, :]
        h_tilde = self.h_tilde[:, t, :]
        h_prev = self.h_states[:, t, :]

        # dL/dh_t -> dL/dz, dL/dh_tilde, dL/dh_prev
        dz = dh * (h_prev - h_tilde)
        dh_tilde = dh * (1 - z)
        dh_prev_from_h = dh * z

        # Backprop through tanh for h_tilde
        dtanh = dh_tilde * (1 - h_tilde**2)

        # dL/dh_tilde -> dL/dW_h, dL/dU_h, dL/dr
        d_W[:, 2*self.units:] += np.dot(self.inputs[:, t, :].T, dtanh)
        d_U[:, 2*self.units:] += np.dot((r * h_prev).T, dtanh)
        d_b[2*self.units:] += np.sum(dtanh, axis=0)

        dr_h_prev = np.dot(dtanh, self.params['U'][:, 2*self.units:].T)
        dr = dr_h_prev * h_prev
        dh_prev_from_tilde = dr_h_prev * r

        # Backprop through sigmoids for z, r
        dz_logits = dz * z * (1 - z)
        dr_logits = dr * r * (1 - r)
        dzr_logits = np.concatenate([dz_logits, dr_logits], axis=1)

        # dL/dzr -> dL/dW_zr, dL/dU_zr
        d_W[:, :2*self.units] += np.dot(self.inputs[:, t, :].T, dzr_logits)
        d_U[:, :2*self.units] += np.dot(h_prev.T, dzr_logits)
        d_b[:2*self.units] += np.sum(dzr_logits, axis=0)

        dh_prev_from_gates = np.dot(dzr_logits, self.params['U'][:, :2*self.units].T)

        # Total dh_prev for next step
        dh_next = dh_prev_from_h + dh_prev_from_tilde + dh_prev_from_gates

        # Gradient wrt inputs
        d_inputs[:, t, :] = \
            np.dot(dzr_logits, self.params['W'][:, :2*self.units].T) + \
            np.dot(dtanh, self.params['W'][:, 2*self.units:].T)

    self.grads['W'] = d_W
    self.grads['U'] = d_U
    self.grads['b'] = d_b
    return d_inputs
```

🧠 **"The backward loop goes in REVERSE order of the forward loop — that's the 'through time' part of BPTT"**

🔍 **Line 88**: `for t in range(timesteps - 1, -1, -1)` — Same reverse loop as SimpleRNN.

🔍 **Line 89**: `dh = grad_output + dh_next` — Gradient from two sources (same as SimpleRNN).

---

**Step 1: Split `dh` into contributions through the three paths of `h_t = (1-z)*h_tilde + z*h_prev`**

The final hidden state formula is:

$$h_t = (1 - z) \cdot \tilde{h}_t + z \cdot h_{t-1}$$

By the product rule, the gradient `dh` splits into three terms:

🔍 **Line 97**: `dz = dh * (h_prev - h_tilde)` — Gradient through `z` in both `(1-z)*h_tilde` and `z*h_prev`:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \cdot (h_{t-1} - \tilde{h}_t)$$

When `h_prev > h_tilde`, increasing `z` (keeping more of old state) reduces loss, and vice versa.

🔍 **Line 98**: `dh_tilde = dh * (1 - z)` — Gradient through `h_tilde` in the `(1-z)` path.

🔍 **Line 99**: `dh_prev_from_h = dh * z` — Gradient through `h_prev` in the `z` path.

📐 All three are shape `(batch, units)`.

---

**Step 2: Backprop through the candidate `h_tilde = tanh(x*Wh + (r*h_prev)*Uh + bh)`**

🔍 **Line 102**: `dtanh = dh_tilde * (1 - h_tilde**2)` — Derivative of tanh.

📐 Shape `(batch, units)`.

🔍 **Lines 106-108**: Gradient w.r.t. the **candidate** (last `units`) portion of the weights:

- `d_W[:, 2*units:] += x_t.T @ dtanh` — `(features, batch) @ (batch, units)` → `(features, units)` = shape of `Wh`.
- `d_U[:, 2*units:] += (r * h_prev).T @ dtanh` — `(units, batch) @ (batch, units)` → `(units, units)` = shape of `Uh`. Note that the reset gate `r` element-wise multiplies `h_prev` before the matrix multiply.
- `d_b[2*units:] += sum(dtanh, axis=0)` — `(units,)`.

---

**Step 3: Backprop through the reset gate `r = sigmoid(x*Wr + h_prev*Ur + br)`**

🔍 **Line 110**: `dr_h_prev = dtanh @ Uh.T` — The gradient w.r.t. `(r * h_prev)`, before the element-wise multiply.

📐 `(batch, units) @ (units, units)` → `(batch, units)`.

🔍 **Line 111**: `dr = dr_h_prev * h_prev` — Gradient w.r.t. `r`. By the product rule of `r * h_prev`:

$$\frac{\partial L}{\partial r} = \frac{\partial L}{\partial (r \cdot h_{prev})} \odot h_{prev}$$

🔍 **Line 112**: `dh_prev_from_tilde = dr_h_prev * r` — The gradient through `r * h_prev` w.r.t. `h_prev`. This is the **other half** of the product rule: if `r` is large, `h_prev` has more influence on the output.

---

**Step 4: Backprop through the sigmoids for `z` and `r`**

🔍 **Lines 115-116**: Sigmoid derivative: `dz_logits = dz * z * (1 - z)` and `dr_logits = dr * r * (1 - r)`. For sigmoid $\sigma(x)$, the derivative is $\sigma(x) \cdot (1 - \sigma(x))$.

🔍 **Line 117**: `dzr_logits = concat([dz_logits, dr_logits])` — Concatenate back into `(batch, 2*units)` for the gate weight updates.

---

**Step 5: Accumulate gradients for the gate weights**

🔍 **Lines 120-122**:
- `d_W[:, :2*units] += x_t.T @ dzr_logits` — `(features, batch) @ (batch, 2*units)` → `(features, 2*units)` = shape of `[Wz | Wr]`.
- `d_U[:, :2*units] += h_prev.T @ dzr_logits` — `(units, batch) @ (batch, 2*units)` → `(units, 2*units)` = shape of `[Uz | Ur]`.
- `d_b[:2*units] += sum(dzr_logits, axis=0)` — `(2*units,)`.

---

**Step 6: Compute `dh_next` — the gradient to pass to the previous timestep**

🔍 **Line 124**: `dh_prev_from_gates = dzr_logits @ U[:, :2*units].T` — Gradient through the gate's recurrent connection.

🔍 **Line 127**: `dh_next = dh_prev_from_h + dh_prev_from_tilde + dh_prev_from_gates` — The total gradient w.r.t. `h_prev` is the sum of three paths:

1. `dh_prev_from_h` — from the `z * h_prev` direct connection
2. `dh_prev_from_tilde` — from the `(r * h_prev)` in the candidate computation
3. `dh_prev_from_gates` — from `h_prev @ U[:, :2*units]` in the gate logits

---

**Step 7: Gradient w.r.t. inputs**

🔍 **Lines 130-131**: `d_inputs[:, t, :]` has two contributions:

1. `dzr_logits @ W[:, :2*units].T` — From the gates (Wz, Wr portions of W)
2. `dtanh @ W[:, 2*units:].T` — From the candidate (Wh portion of W)

📐 Shape: `(batch, 2*units) @ (2*units, features)` and `(batch, units) @ (units, features)` → both `(batch, features)`, added together.

---

## Try it yourself

```python
from neutro.layers.recurrent import SimpleRNN, GRU
import numpy as np

# SimpleRNN
rnn = SimpleRNN(units=64, return_sequences=True)
x = np.random.randn(4, 10, 32)  # (batch, timesteps, features)
y = rnn(x)
print(f"RNN output: {y.shape}")  # (4, 10, 64)

# GRU
gru = GRU(units=64, return_sequences=False)
z = gru(x)
print(f"GRU output: {z.shape}")  # (4, 64)

# Backward
dL_dy = np.random.randn(4, 64)
dL_dx = gru.backward(dL_dy)
print(f"GRU input grad: {dL_dx.shape}")  # (4, 10, 32)
print(f"GRU W grad: {gru.grads['W'].shape}")  # (32, 192)
print(f"GRU U grad: {gru.grads['U'].shape}")  # (64, 192)
```

## What to read next

- **`docs/layers/recurrent/lstm.md`** — The LSTM: four gates, a cell state, and why it solves the vanishing gradient problem.
- **`docs/layers/core/dense.md`** — If you need a refresher on how `build`, `forward`, and `backward` work in simpler layers.
## SimpleRNN

### What does this layer do?

An RNN processes sequences one step at a time, maintaining a hidden state that carries information forward. At each timestep, it combines the current input with the previous hidden state through a learned transformation.

### The math

$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$$

Let's unpack every symbol:

- **$x_t$** — The input at timestep `t`. Shape `(batch, features)`. Each timestep's slice of the input sequence.
- **$W_x$** — Input weight matrix. Shape `(features, units)`. Controls how the current input influences the new hidden state.
- **$h_{t-1}$** — The hidden state from the *previous* timestep. Shape `(batch, units)`. This is the "memory" that carries information across timesteps.
- **$W_h$** — Recurrent weight matrix. Shape `(units, units)`. Controls how the previous hidden state influences the new one.
- **$b$** — Bias vector. Shape `(units,)`.
- **$\tanh$** — Hyperbolic tangent activation, squashing values into `(-1, 1)`.
- **$h_t$** — The new hidden state. Also the output at this timestep (if `return_sequences=True`).

### Walking through the code

#### `__init__` / `build`

```python
def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.return_sequences = return_sequences
    self.activation_name = activation

def build(self, input_shape):
    self.features = input_shape[-1]
    init = get_initializer('glorot_uniform')
    self.params['Wx'] = init((self.features, self.units))
    self.params['Wh'] = init((self.units, self.units))
    self.params['b'] = get_initializer('zeros')((self.units,))
    super().build(input_shape)
```

🔍 **`features = input_shape[-1]`** — We grab the last dimension of the input. If input is `(batch, timesteps, features)`, this is the feature dimension. The first two (batch and timesteps) are handled dynamically.

🔍 **`Wx` shape `(features, units)`** — Maps `features` input dimensions → `units` hidden dimensions. If input is 128-dimensional and we want 64 hidden units, this is `(128, 64)`.

🔍 **`Wh` shape `(units, units)`** — Maps hidden state → hidden state. A square matrix: `(64, 64)`. This is what makes RNNs "recurrent" — the same hidden-to-hidden weight is applied at every timestep.

🔍 **`b` shape `(units,)`** — One bias per hidden unit. Broadcast across batch.

#### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    batch, timesteps, _ = inputs.shape
    self.h_states = np.zeros((batch, timesteps + 1, self.units))

    for t in range(timesteps):
        z = np.dot(inputs[:, t, :], self.params['Wx']) + \
            np.dot(self.h_states[:, t, :], self.params['Wh']) + self.params['b']
        if self.activation_name == 'tanh':
            self.h_states[:, t+1, :] = np.tanh(z)
        else:
            self.h_states[:, t+1, :] = z
    if self.return_sequences:
        return self.h_states[:, 1:, :]
    return self.h_states[:, -1, :]
```

🔍 **Line 27**: `self.inputs = inputs` — Cached for backward. `backward` doesn't receive the original inputs — it only gets `grad_output` — so we must save `inputs` now.

🔍 **Line 28**: `self.h_states = np.zeros((batch, timesteps + 1, self.units))` — We allocate space for **all** hidden states, one per timestep *plus one extra* for the initial `h_0 = 0`. Shape: `(batch, T+1, units)`.

Why `T+1`? So `h_states[:, t, :]` and `h_states[:, t+1, :]` both index validly in the loop, where `t` goes `0` to `T-1`. Entry `0` is `h_0` (all zeros), entry `1` is `h_1`, ..., entry `T` is `h_T`.

🔍 **Line 31**: The core computation:

📐 `inputs[:, t, :]` is `(batch, features)` @ `Wx` `(features, units)` → `(batch, units)`

📐 `h_states[:, t, :]` is `(batch, units)` @ `Wh` `(units, units)` → `(batch, units)`

📐 Adding them plus bias (broadcast) gives `z` of shape `(batch, units)`.

🔍 **Line 32-35**: `np.tanh(z)` applies the squashing non-linearity. The result is stored at position `t+1` in `h_states`, making it `h_t`.

🔍 **Lines 36-38**: If `return_sequences=True`, return all hidden states `h_1` through `h_T` — shape `(batch, T, units)`. If `False`, return only the last hidden state `h_T` — shape `(batch, units)`.

📐 **Output shapes**:
- `return_sequences=True`: `(batch, timesteps, units)`
- `return_sequences=False`: `(batch, units)`

🔍 Why do we cache ALL hidden states (not just the last one)? Because `backward` needs `h_t` and `h_{t+1}` for every `t` to compute gradients. Without caching, we'd have to re-run the forward loop.

#### `backward` (BPTT — Backpropagation Through Time)

```python
def backward(self, grad_output):
    batch, timesteps, _ = self.inputs.shape
    d_Wx, d_Wh, d_b = np.zeros_like(self.params['Wx']), \
                      np.zeros_like(self.params['Wh']), \
                      np.zeros_like(self.params['b'])
    d_inputs = np.zeros_like(self.inputs)
    dh_next = np.zeros((batch, self.units))

    for t in range(timesteps - 1, -1, -1):
        dh = (grad_output[:, t, :] if self.return_sequences \
              else (grad_output if t == timesteps - 1 else 0)) + dh_next
        dz = dh * (1 - self.h_states[:, t+1, :]**2)
        d_Wx += np.dot(self.inputs[:, t, :].T, dz)
        d_Wh += np.dot(self.h_states[:, t, :].T, dz)
        d_b += np.sum(dz, axis=0)
        d_inputs[:, t, :] = np.dot(dz, self.params['Wx'].T)
        dh_next = np.dot(dz, self.params['Wh'].T)

    self.grads['Wx'], self.grads['Wh'], self.grads['b'] = d_Wx, d_Wh, d_b
    return d_inputs
```

🧠 **"The backward loop goes in REVERSE order of the forward loop — that's the 'through time' part of BPTT"**

🔍 **Line 46**: `for t in range(timesteps - 1, -1, -1)` — Loop from `T-1` down to `0`. The forward loop went `0, 1, 2, ..., T-1`. The backward loop goes `T-1, ..., 2, 1, 0`.

🔍 **Line 47**: `dh = grad_output + dh_next` — The gradient arriving at this timestep has **two sources**:

1. **From above**: the upstream gradient `grad_output` from the loss (or next layer). If `return_sequences=True`, each timestep `t` gets its own slice `grad_output[:, t, :]`. If `return_sequences=False`, only the last timestep gets the gradient; all others get a zero contribution from this source.

2. **From the future**: `dh_next` — the gradient flowing *back* from timestep `t+1`. This is computed on the **previous iteration** of the backward loop (which was timestep `t+1`, since we're going in reverse).

This two-source pattern is the essence of BPTT — future timesteps send gradient information backward through `Wh`.

🔍 **Line 48**: `dz = dh * (1 - h_{t+1}^2)` — This is the derivative of $\tanh$: $\frac{d}{dz}\tanh(z) = 1 - \tanh(z)^2 = 1 - h_{t+1}^2$.

We multiply `dh` by this derivative (element-wise) to backpropagate through the tanh activation. This is the chain rule: $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z}$.

🔍 **Lines 49-51**: Accumulate gradients for each parameter:

- `d_Wx += x_t^T @ dz` — Shape: `(features, batch) @ (batch, units)` → `(features, units)` = `Wx` shape.
- `d_Wh += h_t^T @ dz` — Shape: `(units, batch) @ (batch, units)` → `(units, units)` = `Wh` shape.
- `d_b += sum(dz, axis=0)` — Sum over batch → `(units,)` = `b` shape.

These are **accumulated** (with `+=`) across all timesteps because the same `Wx`, `Wh`, and `b` are used at every timestep — the total gradient is the sum of the gradients from each timestep.

🔍 **Line 52**: `d_inputs[:, t, :] = dz @ Wx.T` — The gradient w.r.t. the input at this timestep. Shape: `(batch, units) @ (units, features)` → `(batch, features)`.

🔍 **Line 53**: `dh_next = dz @ Wh.T` — The gradient to pass to the previous timestep. Shape: `(batch, units) @ (units, units)` → `(batch, units)`. This becomes `dh_next` in the `t-1` iteration.

📐 Gradient shapes flowing backward:

```
t = T-1:
  grad_output  →  dh (batch, units)  →  dz (batch, units)
                                         → d_Wx (features, units)
                                         → d_Wh (units, units)
                                         → d_b (units,)
                                         → d_inputs[:, T-1, :] (batch, features)
                                         → dh_next (batch, units)
                                                         ↓
t = T-2:                                    dh_next passed to previous step
  grad_output  +  dh_next  →  dh  →  dz  → ... same pattern
```

---

## GRU

### What does this layer do?

A Gated Recurrent Unit (GRU) is a simplified version of LSTM that merges the cell state and hidden state. It uses two gates — an **update gate** (how much to keep vs. replace) and a **reset gate** (how much to forget the past) — to control information flow.

### The math

$$z_t = \sigma(x_t W_z + h_{t-1} U_z) \quad \text{(update gate)}$$

$$r_t = \sigma(x_t W_r + h_{t-1} U_r) \quad \text{(reset gate)}$$

$$\tilde{h}_t = \tanh(x_t W_h + (r_t \odot h_{t-1}) U_h) \quad \text{(candidate hidden state)}$$

$$h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1} \quad \text{(final hidden state)}$$

Let's unpack every symbol:

- **$z_t$** — Update gate. Values in `(0, 1)`. A value near 1 means "keep the old state"; near 0 means "replace with the candidate." Note: the naming convention is sometimes flipped — here $z$ controls how much of the *old* state to keep, so $(1-z)$ controls how much of the *new* candidate to take.
- **$r_t$** — Reset gate. Values in `(0, 1)`. Controls how much of the past hidden state to forget when computing the candidate.
- **$\tilde{h}_t$** — Candidate hidden state. Like a regular RNN's hidden state, but modulated by the reset gate.
- **$W_z, W_r, W_h$** — Input weight matrices. Each has shape `(features, units)`. They are stacked into one big matrix `W` of shape `(features, 3 * units)` for efficiency.
- **$U_z, U_r, U_h$** — Recurrent weight matrices. Each has shape `(units, units)`. Stacked into `U` of shape `(units, 3 * units)`.
- **$h_t$** — Final hidden state. An interpolation between the old state and the candidate, controlled by `z`.

### Walking through the code

#### `__init__` / `build`

```python
def __init__(self, units, return_sequences=False, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.return_sequences = return_sequences

def build(self, input_shape):
    self.features = input_shape[-1]
    init = get_initializer('glorot_uniform')
    self.params['W'] = init((self.features, 3 * self.units))
    self.params['U'] = init((self.units, 3 * self.units))
    self.params['b'] = get_initializer('zeros')((3 * self.units,))
    super().build(input_shape)
```

🔍 **Weight organization**: Instead of 6 separate matrices (Wz, Wr, Wh, Uz, Ur, Uh), the GRU concatenates them:

```
W = [Wz | Wr | Wh]    shape: (features, 3*units)
U = [Uz | Ur | Uh]    shape: (units, 3*units)
b = [bz | br | bh]    shape: (3*units,)
```

The first `2*units` columns are for the update and reset gates (z, r). The last `units` columns are for the candidate hidden state (h_tilde).

📐 **Why concatenate?** A single matrix multiply `x @ W` is faster than three separate ones. We split the result afterward.

#### `forward`

```python
def forward(self, inputs, training=False):
    self.inputs = inputs
    batch, timesteps, _ = inputs.shape
    self.h_states = np.zeros((batch, timesteps + 1, self.units))

    self.z_gates = np.zeros((batch, timesteps, self.units))
    self.r_gates = np.zeros((batch, timesteps, self.units))
    self.h_tilde = np.zeros((batch, timesteps, self.units))
    self.x_W = np.dot(inputs, self.params['W'])  # (batch, timesteps, 3*units)

    for t in range(timesteps):
        x_W_t = self.x_W[:, t, :] + self.params['b']
        h_prev = self.h_states[:, t, :]

        z_r_hidden = np.dot(h_prev, self.params['U'][:, :2*self.units])
        z_r_logits = x_W_t[:, :2*self.units] + z_r_hidden
        z_r = self._sigmoid(z_r_logits)

        z = z_r[:, :self.units]
        r = z_r[:, self.units:]

        self.z_gates[:, t, :] = z
        self.r_gates[:, t, :] = r

        h_tilde_hidden = np.dot(r * h_prev, self.params['U'][:, 2*self.units:])
        h_tilde_logits = x_W_t[:, 2*self.units:] + h_tilde_hidden
        h_tilde = np.tanh(h_tilde_logits)
        self.h_tilde[:, t, :] = h_tilde

        self.h_states[:, t+1, :] = (1 - z) * h_tilde + z * h_prev

    if self.return_sequences:
        return self.h_states[:, 1:, :]
    return self.h_states[:, -1, :]
```

🔍 **Lines 47-49**: Pre-allocate storage for **all** intermediate values — `z_gates`, `r_gates`, `h_tilde` across all timesteps. These are cached for the backward pass.

🔍 **Line 50**: `self.x_W = np.dot(inputs, self.params['W'])` — Compute the input-to-hidden projection for **all timesteps at once**.

📐 `inputs` is `(batch, timesteps, features)` @ `W` is `(features, 3*units)` → `(batch, timesteps, 3*units)`.

This is an optimization: one big matrix multiply instead of `T` small ones. At timestep `t`, we slice `self.x_W[:, t, :]`.

🔍 **Lines 57-62**: Compute the update gate `z` and reset gate `r`:

1. `z_r_hidden = h_prev @ U[:, :2*units]` — The recurrent part of the gates.
2. `z_r_logits = x_W_t[:, :2*units] + z_r_hidden` — Add the input part (from the pre-computed `x_W`).
3. `z_r = sigmoid(z_r_logits)` — Both gates share one sigmoid computation.
4. Split: `z = z_r[:, :units]`, `r = z_r[:, units:]`.

🔍 **Lines 64-71**: Compute the candidate hidden state:

1. `h_tilde_hidden = (r * h_prev) @ U[:, 2*units:]` — The reset gate element-wise multiplies the previous hidden state, zeroing out some dimensions.
2. `h_tilde_logits = x_W_t[:, 2*units:] + h_tilde_hidden` — Add the input part.
3. `h_tilde = tanh(h_tilde_logits)` — Squash to `(-1, 1)`.

🔍 **Line 74**: `h_t = (1 - z) * h_tilde + z * h_prev` — The final hidden state is an **interpolation** between the old state and the candidate. When `z` is close to 1, we mostly keep the old state. When `z` is close to 0, we mostly take the new candidate.

📐 **Why cache `z_gates`, `r_gates`, `h_tilde`, and `x_W`?** Because `backward` needs every gate value at every timestep, plus the input projection. Without caching them all, backward would have to recompute the entire forward pass.

#### `backward` (BPTT)

```python
def backward(self, grad_output):
    batch, timesteps, _ = self.inputs.shape
    d_W = np.zeros_like(self.params['W'])
    d_U = np.zeros_like(self.params['U'])
    d_b = np.zeros_like(self.params['b'])
    d_inputs = np.zeros_like(self.inputs)
    dh_next = np.zeros((batch, self.units))

    for t in range(timesteps - 1, -1, -1):
        dh = (grad_output[:, t, :] if self.return_sequences
              else (grad_output if t == timesteps - 1 else 0)) + dh_next

        z = self.z_gates[:, t, :]
        r = self.r_gates[:, t, :]
        h_tilde = self.h_tilde[:, t, :]
        h_prev = self.h_states[:, t, :]

        # dL/dh_t -> dL/dz, dL/dh_tilde, dL/dh_prev
        dz = dh * (h_prev - h_tilde)
        dh_tilde = dh * (1 - z)
        dh_prev_from_h = dh * z

        # Backprop through tanh for h_tilde
        dtanh = dh_tilde * (1 - h_tilde**2)

        # dL/dh_tilde -> dL/dW_h, dL/dU_h, dL/dr
        d_W[:, 2*self.units:] += np.dot(self.inputs[:, t, :].T, dtanh)
        d_U[:, 2*self.units:] += np.dot((r * h_prev).T, dtanh)
        d_b[2*self.units:] += np.sum(dtanh, axis=0)

        dr_h_prev = np.dot(dtanh, self.params['U'][:, 2*self.units:].T)
        dr = dr_h_prev * h_prev
        dh_prev_from_tilde = dr_h_prev * r

        # Backprop through sigmoids for z, r
        dz_logits = dz * z * (1 - z)
        dr_logits = dr * r * (1 - r)
        dzr_logits = np.concatenate([dz_logits, dr_logits], axis=1)

        # dL/dzr -> dL/dW_zr, dL/dU_zr
        d_W[:, :2*self.units] += np.dot(self.inputs[:, t, :].T, dzr_logits)
        d_U[:, :2*self.units] += np.dot(h_prev.T, dzr_logits)
        d_b[:2*self.units] += np.sum(dzr_logits, axis=0)

        dh_prev_from_gates = np.dot(dzr_logits, self.params['U'][:, :2*self.units].T)

        # Total dh_prev for next step
        dh_next = dh_prev_from_h + dh_prev_from_tilde + dh_prev_from_gates

        # Gradient wrt inputs
        d_inputs[:, t, :] = \
            np.dot(dzr_logits, self.params['W'][:, :2*self.units].T) + \
            np.dot(dtanh, self.params['W'][:, 2*self.units:].T)

    self.grads['W'] = d_W
    self.grads['U'] = d_U
    self.grads['b'] = d_b
    return d_inputs
```

🧠 **"The backward loop goes in REVERSE order of the forward loop — that's the 'through time' part of BPTT"**

🔍 **Line 88**: `for t in range(timesteps - 1, -1, -1)` — Same reverse loop as SimpleRNN.

🔍 **Line 89**: `dh = grad_output + dh_next` — Gradient from two sources (same as SimpleRNN).

---

**Step 1: Split `dh` into contributions through the three paths of `h_t = (1-z)*h_tilde + z*h_prev`**

The final hidden state formula is:

$$h_t = (1 - z) \cdot \tilde{h}_t + z \cdot h_{t-1}$$

By the product rule, the gradient `dh` splits into three terms:

🔍 **Line 97**: `dz = dh * (h_prev - h_tilde)` — Gradient through `z` in both `(1-z)*h_tilde` and `z*h_prev`:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \cdot (h_{t-1} - \tilde{h}_t)$$

When `h_prev > h_tilde`, increasing `z` (keeping more of old state) reduces loss, and vice versa.

🔍 **Line 98**: `dh_tilde = dh * (1 - z)` — Gradient through `h_tilde` in the `(1-z)` path.

🔍 **Line 99**: `dh_prev_from_h = dh * z` — Gradient through `h_prev` in the `z` path.

📐 All three are shape `(batch, units)`.

---

**Step 2: Backprop through the candidate `h_tilde = tanh(x*Wh + (r*h_prev)*Uh + bh)`**

🔍 **Line 102**: `dtanh = dh_tilde * (1 - h_tilde**2)` — Derivative of tanh.

📐 Shape `(batch, units)`.

🔍 **Lines 106-108**: Gradient w.r.t. the **candidate** (last `units`) portion of the weights:

- `d_W[:, 2*units:] += x_t.T @ dtanh` — `(features, batch) @ (batch, units)` → `(features, units)` = shape of `Wh`.
- `d_U[:, 2*units:] += (r * h_prev).T @ dtanh` — `(units, batch) @ (batch, units)` → `(units, units)` = shape of `Uh`. Note that the reset gate `r` element-wise multiplies `h_prev` before the matrix multiply.
- `d_b[2*units:] += sum(dtanh, axis=0)` — `(units,)`.

---

**Step 3: Backprop through the reset gate `r = sigmoid(x*Wr + h_prev*Ur + br)`**

🔍 **Line 110**: `dr_h_prev = dtanh @ Uh.T` — The gradient w.r.t. `(r * h_prev)`, before the element-wise multiply.

📐 `(batch, units) @ (units, units)` → `(batch, units)`.

🔍 **Line 111**: `dr = dr_h_prev * h_prev` — Gradient w.r.t. `r`. By the product rule of `r * h_prev`:

$$\frac{\partial L}{\partial r} = \frac{\partial L}{\partial (r \cdot h_{prev})} \odot h_{prev}$$

🔍 **Line 112**: `dh_prev_from_tilde = dr_h_prev * r` — The gradient through `r * h_prev` w.r.t. `h_prev`. This is the **other half** of the product rule: if `r` is large, `h_prev` has more influence on the output.

---

**Step 4: Backprop through the sigmoids for `z` and `r`**

🔍 **Lines 115-116**: Sigmoid derivative: `dz_logits = dz * z * (1 - z)` and `dr_logits = dr * r * (1 - r)`. For sigmoid $\sigma(x)$, the derivative is $\sigma(x) \cdot (1 - \sigma(x))$.

🔍 **Line 117**: `dzr_logits = concat([dz_logits, dr_logits])` — Concatenate back into `(batch, 2*units)` for the gate weight updates.

---

**Step 5: Accumulate gradients for the gate weights**

🔍 **Lines 120-122**:
- `d_W[:, :2*units] += x_t.T @ dzr_logits` — `(features, batch) @ (batch, 2*units)` → `(features, 2*units)` = shape of `[Wz | Wr]`.
- `d_U[:, :2*units] += h_prev.T @ dzr_logits` — `(units, batch) @ (batch, 2*units)` → `(units, 2*units)` = shape of `[Uz | Ur]`.
- `d_b[:2*units] += sum(dzr_logits, axis=0)` — `(2*units,)`.

---

**Step 6: Compute `dh_next` — the gradient to pass to the previous timestep**

🔍 **Line 124**: `dh_prev_from_gates = dzr_logits @ U[:, :2*units].T` — Gradient through the gate's recurrent connection.

🔍 **Line 127**: `dh_next = dh_prev_from_h + dh_prev_from_tilde + dh_prev_from_gates` — The total gradient w.r.t. `h_prev` is the sum of three paths:

1. `dh_prev_from_h` — from the `z * h_prev` direct connection
2. `dh_prev_from_tilde` — from the `(r * h_prev)` in the candidate computation
3. `dh_prev_from_gates` — from `h_prev @ U[:, :2*units]` in the gate logits

---

**Step 7: Gradient w.r.t. inputs**

🔍 **Lines 130-131**: `d_inputs[:, t, :]` has two contributions:

1. `dzr_logits @ W[:, :2*units].T` — From the gates (Wz, Wr portions of W)
2. `dtanh @ W[:, 2*units:].T` — From the candidate (Wh portion of W)

📐 Shape: `(batch, 2*units) @ (2*units, features)` and `(batch, units) @ (units, features)` → both `(batch, features)`, added together.

---

## Try it yourself

```python
from neutro.layers.recurrent import SimpleRNN, GRU
import numpy as np

# SimpleRNN
rnn = SimpleRNN(units=64, return_sequences=True)
x = np.random.randn(4, 10, 32)  # (batch, timesteps, features)
y = rnn(x)
print(f"RNN output: {y.shape}")  # (4, 10, 64)

# GRU
gru = GRU(units=64, return_sequences=False)
z = gru(x)
print(f"GRU output: {z.shape}")  # (4, 64)

# Backward
dL_dy = np.random.randn(4, 64)
dL_dx = gru.backward(dL_dy)
print(f"GRU input grad: {dL_dx.shape}")  # (4, 10, 32)
print(f"GRU W grad: {gru.grads['W'].shape}")  # (32, 192)
print(f"GRU U grad: {gru.grads['U'].shape}")  # (64, 192)
```

## What to read next

- **`docs/layers/recurrent/lstm.md`** — The LSTM: four gates, a cell state, and why it solves the vanishing gradient problem.
- **`docs/layers/core/dense.md`** — If you need a refresher on how `build`, `forward`, and `backward` work in simpler layers.
