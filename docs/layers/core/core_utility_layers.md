# Core Utility Layers

## Dropout — `neutro/layers/core/dropout.py`

### What does this layer do?

Dropout randomly "drops" (sets to zero) a fraction of neurons during training. This forces the network to not rely too heavily on any single neuron, which prevents **co-adaptation** and acts as a regularization technique. During inference, all neurons are used at full strength.

### The math, in plain English

$$
y = \begin{cases} \frac{m \odot x}{1 - p} & \text{during training} \\ x & \text{during inference} \end{cases}
$$

Each element of the mask $m$ is drawn from a Bernoulli distribution with probability $1-p$ (i.e., it is 1 with probability $1-p$, and 0 with probability $p$). The symbol $\odot$ means element-wise multiplication.

**Why do we divide by $1-p$?** — If a neuron is kept with probability $1-p$, its expected value during training is $x \cdot (1-p)$. By dividing by $1-p$, we restore the expected magnitude to $x$, so the training and inference outputs have the same scale. Without this scaling, the network would see much larger values during inference and produce wrong results.

### Walking through the code

#### `__init__`

```python
def __init__(self, rate, **kwargs):
    super().__init__(**kwargs)
    self.rate = rate
    self.mask = None
```

🔍 **Line `self.mask = None`**: The mask is created during `forward` and cached on `self` so that `backward` can reuse it. Before the first forward pass, it starts as `None`.

#### `forward`

```python
def forward(self, inputs, training=False):
    if not training or self.rate == 0:
        return inputs
    self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
    return inputs * self.mask
```

🔍 **Line `if not training or self.rate == 0`**: Dropout only applies during training. At inference time, we return the input unchanged. If `rate=0`, nothing is dropped, so we skip the overhead.

🔍 **Line `self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)`**: This is the core. `np.random.binomial(1, 1-self.rate, ...)` creates a binary mask where each element is 1 with probability $1-p$. By dividing by $1-p$ right here in the mask, the subsequent multiplication `inputs * self.mask` naturally produces the scaled output. The mask is cached for the backward pass.

📐 **Shape**: `mask.shape == inputs.shape` — every element gets its own independent mask value.

🔍 **Line `return inputs * self.mask`**: Element-wise multiplication applies the mask.

#### `backward`

```python
def backward(self, grad_output):
    if self.mask is None:
        return grad_output
    return grad_output * self.mask
```

🔍 **Line `grad_output * self.mask`**: The derivative of $y = m \odot x$ with respect to $x$ is $m$ (the mask). So the gradient is simply multiplied by the same mask used in the forward pass. Note the mask already includes the $1/(1-p)$ scaling factor.

🔍 **Line `if self.mask is None`**: If we never called forward (or called it with `training=False`), there's no mask. In that case, the gradient passes through unchanged — just like the forward pass.

---

## Flatten — `neutro/layers/core/flatten.py`

### What does this layer do?

Flatten reshapes a multi-dimensional input (e.g., a batch of images) into a 2D array where each sample becomes a single 1D vector. This is typically used between convolutional layers and dense layers.

### The math, in plain English

$$
\text{Input shape: } (N, d_1, d_2, \dots, d_k) \quad \Longrightarrow \quad \text{Output shape: } (N, d_1 \times d_2 \times \dots \times d_k)
$$

The batch dimension $N$ is preserved. All other dimensions are multiplied together to form a single feature dimension.

### Walking through the code

#### `__init__`

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
```

No special parameters — flattening is purely a shape transformation.

#### `build`

```python
def build(self, input_shape):
    self.input_shape_orig = input_shape
    super().build(input_shape)
```

🔍 **Line `self.input_shape_orig = input_shape`**: We save the original input shape so that `backward` knows how to "un-flatten" the gradient.

#### `compute_output_shape`

```python
def compute_output_shape(self, input_shape):
    import numpy as np
    return (input_shape[0], int(np.prod(input_shape[1:])))
```

📐 `input_shape[0]` is the batch dimension (kept as-is). `int(np.prod(input_shape[1:]))` multiplies all remaining dimensions together. For example, `(None, 28, 28, 1)` → `(None, 784)`.

#### `forward`

```python
def forward(self, inputs, training=False):
    self.input_shape_orig = inputs.shape
    return inputs.reshape(inputs.shape[0], -1)
```

🔍 **Line `self.input_shape_orig = inputs.shape`**: We cache the **actual** shape from the real data (not the symbolic shape from `build`). This is critical because `build` receives symbolic shapes (e.g., `(None, 28, 28)`), but we need the concrete batch size.

📐 **Shape**: `inputs.shape` = `(batch, d1, d2, ..., dk)`. `inputs.shape[0]` is the batch size. `-1` tells NumPy to infer the remaining dimension as the product of all other dimensions. So `(8, 4, 4, 16)` → `(8, 256)`.

#### `backward`

```python
def backward(self, grad_output):
    return grad_output.reshape(self.input_shape_orig)
```

🔍 **Line `grad_output.reshape(self.input_shape_orig)`**: The backward pass simply reverses the flattening. The gradient is reshaped back to the original multi-dimensional shape. This works because `reshape` is a purely geometric operation — the gradient flows through each element along the path of the reshape.

---

## MoE Layer (Mixture of Experts) — `neutro/layers/core/moe.py`

### What does this layer do?

A Mixture-of-Experts (MoE) layer maintains multiple "expert" sub-networks and a **router** that decides which expert(s) should process each input token. Instead of every token passing through every expert (expensive), each token only activates the top-$k$ most relevant experts. This scales model capacity without proportionally scaling compute.

### The math, in plain English

$$
y_t = \sum_{i=1}^{E} g_i(x_t) \cdot E_i(x_t)
$$

For a single token $x_t$:
- $E_i(x)$ is the output of expert $i$ (a small MLP).
- $g_i(x)$ is the router's **gating weight** for expert $i$ — but it is zero for all experts not in the top-$k$.
- The final output is a weighted sum of only the chosen experts' outputs.

The router is just a linear layer followed by softmax:

$$
g(x) = \text{softmax}(x W_{\text{router}})
$$

### Walking through the code

#### `__init__`

```python
def __init__(self, num_experts, top_k, expert_units, **kwargs):
    super().__init__(**kwargs)
    self.num_experts = num_experts
    self.top_k = top_k
    self.expert_units = expert_units
    self.experts = []
```

🔍 **`num_experts`**: The total number of expert MLPs in the pool.

🔍 **`top_k`**: How many experts are activated per token. If `top_k=2`, each token is processed by only 2 out of `num_experts` experts.

🔍 **`expert_units`**: The hidden dimension inside each expert's MLP.

🔍 **`self.experts = []`**: Will be populated in `build` with pairs of Dense layers (one per expert).

#### `build`

```python
def build(self, input_shape):
    self.input_dim = input_shape[-1]

    self.params['router_weight'] = np.random.normal(0, 0.02, (self.input_dim, self.num_experts))

    for i in range(self.num_experts):
        e1 = Dense(self.expert_units, activation='relu')
        e1.build(input_shape)

        expert_shape = list(input_shape)
        expert_shape[-1] = self.expert_units

        e2 = Dense(self.input_dim)
        e2.build(tuple(expert_shape))

        self.experts.append([e1, e2])

    super().build(input_shape)
```

🔍 **Line `self.params['router_weight']`**: The router is a weight matrix of shape `(input_dim, num_experts)`. It maps each input token to a score for each expert.

📐 **Shape**: `router_weight` = `(input_dim, num_experts)` — one column per expert.

🔍 **Each expert is two Dense layers**: `e1` projects from `input_dim` → `expert_units` (with ReLU), and `e2` projects back from `expert_units` → `input_dim`. This is a bottle-neck MLP.

🔍 **`super().build(input_shape)`**: Marks the layer as `built = True`.

#### `compute_output_shape`

```python
def compute_output_shape(self, input_shape):
    return input_shape
```

The output has the same shape as the input — each token's representation is transformed but its dimensionality stays the same.

#### `forward`

```python
def forward(self, x, training=False):
    self.x_shape = x.shape
    self.x_flat = x.reshape(-1, self.input_dim)
    num_tokens = self.x_flat.shape[0]

    # 1. Routing scores
    router_logits = self.x_flat @ self.params['router_weight']

    # Softmax to get probabilities
    router_probs = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
    router_probs /= np.sum(router_probs, axis=-1, keepdims=True)
    self.router_probs = router_probs

    # 2. Select top-k experts
    top_k_indices = np.argsort(router_probs, axis=-1)[:, -self.top_k:]
    self.top_k_indices = top_k_indices

    # 3. Dispatch to experts and combine results
    final_output = np.zeros_like(self.x_flat)
    self.expert_outputs = {}

    for expert_idx in range(self.num_experts):
        token_indices, _ = np.where(top_k_indices == expert_idx)
        if len(token_indices) == 0:
            continue

        tokens = self.x_flat[token_indices]

        out = tokens
        for layer in self.experts[expert_idx]:
            out = layer(out, training=training)

        self.expert_outputs[expert_idx] = (token_indices, out)

        weights = router_probs[token_indices, expert_idx].reshape(-1, 1)
        final_output[token_indices] += weights * out

    return final_output.reshape(self.x_shape)
```

🔍 **Line `self.x_flat = x.reshape(-1, self.input_dim)`**: Flatten the input to 2D: `(batch * seq_len, input_dim)` if the input is 3D, or `(batch, input_dim)` if already 2D. This lets us process each token independently.

📐 **Shape**: `x` = `(batch, seq_len, dim)` → `x_flat` = `(batch * seq_len, dim)`.

🔍 **Lines `router_logits = self.x_flat @ self.params['router_weight']`**: A simple matrix multiplication. Each token gets `num_experts` scores (logits).

📐 **Shape**: `x_flat` is `(T, input_dim)`, `router_weight` is `(input_dim, num_experts)`. Result: `(T, num_experts)`.

🔍 **Lines `router_probs = np.exp(router_logits - ...) ... /= np.sum(...)`**: A numerically stable softmax. We subtract the max logit before exponentiating to prevent overflow.

📐 **Shape**: `router_probs` = `(T, num_experts)` — probabilities summing to 1 per token.

🔍 **Line `top_k_indices = np.argsort(router_probs, axis=-1)[:, -self.top_k:]`**: Sort the probabilities and take the last `top_k` indices (highest scores).

📐 **Shape**: `top_k_indices` = `(T, top_k)` — for each token, which experts to use.

🔍 **Line `token_indices, _ = np.where(top_k_indices == expert_idx)`**: For each expert, find which tokens selected it in their top-k. `np.where` returns the row (token) indices where this expert appears.

🔍 **Line `out = tokens; for layer in self.experts[expert_idx]: out = layer(out, training=training)`**: Pass the selected tokens through the expert's MLP (ReLU then output projection).

🔍 **Line `weights = router_probs[token_indices, expert_idx].reshape(-1, 1)`**: Look up the router probability for this expert for each selected token.

🔍 **Line `final_output[token_indices] += weights * out`**: Weight the expert's output by the router probability and add to the final result. Tokens not assigned to this expert get no contribution.

📐 **Shape**: `weights` is `(num_selected, 1)`, `out` is `(num_selected, input_dim)`. Broadcasting multiplies each row of `out` by its weight.

#### `backward`

```python
def backward(self, grad_output):
    grad_flat = grad_output.reshape(-1, self.input_dim)
    num_tokens = self.x_flat.shape[0]

    dx_flat = np.zeros_like(self.x_flat)
    drouter_logits = np.zeros_like(self.router_probs)

    # 1. Backprop through experts and router probabilities
    for expert_idx, (token_indices, out) in self.expert_outputs.items():
        weights = self.router_probs[token_indices, expert_idx].reshape(-1, 1)
        expert_grad = weights * grad_flat[token_indices]

        drouter_probs_expert = np.sum(grad_flat[token_indices] * out, axis=-1)
        drouter_logits[token_indices, expert_idx] = drouter_probs_expert

        curr_grad = expert_grad
        for layer in reversed(self.experts[expert_idx]):
            curr_grad = layer.backward(curr_grad)

        dx_flat[token_indices] += curr_grad

    # 2. Backprop through Softmax for router
    drouter_logits = self.router_probs * (
        drouter_logits - np.sum(self.router_probs * drouter_logits, axis=-1, keepdims=True)
    )

    # 3. Router weight gradient
    self.grads['router_weight'] = self.x_flat.T @ drouter_logits

    # 4. Add router's contribution to dx
    dx_flat += drouter_logits @ self.params['router_weight'].T

    return dx_flat.reshape(self.x_shape)
```

🔍 **Line `expert_grad = weights * grad_flat[token_indices]`**: The gradient through the weighted combination $y = w \cdot E(x)$. Since $dy/dE = w$, we multiply the upstream gradient by the router weight.

🔍 **Line `drouter_probs_expert = np.sum(grad_flat[token_indices] * out, axis=-1)`**: For each token assigned to this expert, the gradient w.r.t. the router probability weight $w$ is: $dL/dw = dL/dy \cdot E(x)$. We sum across the feature dimension because each token has a single scalar weight per expert.

🔍 **Line `for layer in reversed(self.experts[expert_idx]): curr_grad = layer.backward(curr_grad)`**: Backprop through the expert's MLP layers in reverse order. Each Dense layer computes gradients for its own parameters and passes gradients backward.

🔍 **Line `drouter_logits = self.router_probs * (drouter_logits - np.sum(...))`**: Standard softmax backward pass. For softmax $p_i = e^{z_i} / \sum e^{z_j}$, the Jacobian is $\partial p_i / \partial z_j = p_i (\delta_{ij} - p_j)$. In vectorized form: $dL/dz = p \odot (dL/dp - \sum(p \odot dL/dp))$.

🔍 **Line `self.grads['router_weight'] = self.x_flat.T @ drouter_logits`**: Gradient for the router weight matrix: $dL/dW = x^T \cdot dL/dz$ (outer product).

🔍 **Line `dx_flat += drouter_logits @ self.params['router_weight'].T`**: The router also contributes to the input gradient. $x$ flows into both the router (through the linear layer) and the experts. The total gradient is the sum of both contributions.

---

## Reparameterization — `neutro/layers/core/reparameterization.py`

### What does this layer do?

This layer implements the **reparameterization trick** used in Variational Autoencoders (VAEs). It takes two tensors — a mean $\mu$ and a log-variance $\log \sigma^2$ — and produces a sample from the corresponding Gaussian distribution. The trick is that the sampling operation is rewritten so that gradients can flow through it.

### The math, in plain English

The standard sampling from a Gaussian is $z \sim \mathcal{N}(\mu, \sigma^2)$. But sampling is a stochastic operation with no gradient. The reparameterization trick rewrites it as:

$$
z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)
$$

Here $\sigma = \exp(0.5 \cdot \log \sigma^2) = \sqrt{\sigma^2}$, and $\epsilon$ is a random noise vector drawn from the standard normal. The sampling is now **deterministic given $\epsilon$**, so gradients can flow backward through $\mu$ and $\sigma$ (and to $\log \sigma^2$).

### Walking through the code

#### `__init__`

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
```

No special parameters — the layer just needs to know the shape of the latent space, which comes from the input.

#### `compute_output_shape`

```python
def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
        return input_shape[0]
    return input_shape
```

The output shape is the same as the mean tensor's shape (the first input). The second input (log variance) has the same shape.

#### `forward`

```python
def forward(self, inputs, training=False):
    self.z_mean, self.z_log_var = inputs

    if not training:
        return self.z_mean

    self.epsilon = np.random.normal(size=self.z_mean.shape)
    self.z = self.z_mean + np.exp(0.5 * self.z_log_var) * self.epsilon
    return self.z
```

🔍 **Line `self.z_mean, self.z_log_var = inputs`**: The layer receives **two tensors** as a list: `[z_mean, z_log_var]`. They must have the same shape.

🔍 **Line `if not training: return self.z_mean`**: During inference, we don't sample — we return the mean directly. This gives a deterministic output.

🔍 **Line `self.epsilon = np.random.normal(size=self.z_mean.shape)`**: Draw random noise from the standard normal distribution $\mathcal{N}(0, 1)$. This is cached for the backward pass.

📐 **Shape**: `epsilon.shape == z_mean.shape` — one noise value per element.

🔍 **Line `self.z = self.z_mean + np.exp(0.5 * self.z_log_var) * self.epsilon`**: The reparameterization formula:
- `np.exp(0.5 * self.z_log_var)` computes $\sigma = \sqrt{\sigma^2}$.
- Multiply by $\epsilon$ gives the stochastic part.
- Add $\mu$ to center it.

📐 **Shape**: All tensors have the same shape, e.g., `(batch, latent_dim)`.

🔍 **Line `self.z` is cached** so the backward pass doesn't need to recompute it (though backward uses `z_log_var` and `epsilon`, not `z` directly).

#### `backward`

```python
def backward(self, grad_output):
    grad_mean = grad_output
    grad_log_var = grad_output * np.exp(0.5 * self.z_log_var) * 0.5 * self.epsilon

    return [grad_mean, grad_log_var]
```

🔍 **Line `grad_mean = grad_output`**: The derivative of $z$ with respect to $\mu$ is 1 (from $z = \mu + \sigma \epsilon$). So the gradient passes through unchanged.

🔍 **Line `grad_log_var = grad_output * np.exp(0.5 * self.z_log_var) * 0.5 * self.epsilon`**: Chain rule in action:

$$
\frac{\partial z}{\partial (\log \sigma^2)} = \frac{\partial}{\partial (\log \sigma^2)} \left( \exp(0.5 \cdot \log \sigma^2) \cdot \epsilon \right) = \exp(0.5 \cdot \log \sigma^2) \cdot 0.5 \cdot \epsilon
$$

Breaking it down:
1. Let $a = 0.5 \cdot \log \sigma^2$.
2. Let $b = \exp(a)$ (which is $\sigma$).
3. $z = \mu + b \cdot \epsilon$.
4. $db/da = \exp(a) = b$.
5. $da/d(\log \sigma^2) = 0.5$.
6. So $dz/d(\log \sigma^2) = b \cdot 0.5 \cdot \epsilon = \exp(0.5 \cdot \log \sigma^2) \cdot 0.5 \cdot \epsilon$.

🔍 **The function returns `[grad_mean, grad_log_var]`**: Since the forward received a list of two tensors, the backward must return a list of two gradients — one for each input.

## References

- Srivastava, N., et al. (2014). **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**. *JMLR*.
- Shazeer, N., et al. (2017). **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Kingma, D. P., & Welling, M. (2013). **Auto-Encoding Variational Bayes**. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
