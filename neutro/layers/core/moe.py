import numpy as np

from ..base import Layer
from .dense import Dense
from neutro.autograd import Tensor, ops as autograd_ops


def _make_relu_expert(input_dim, expert_units):
    # Built lazily in build(); placeholder factory.
    return [('relu', expert_units)]


class MoELayer(Layer):
    """Mixture of Experts with top-k routing.

    Dense (all-expert) evaluation for educational clarity: every expert
    runs on every token, then outputs are combined with top-k masked
    router weights. Differentiable via GradientTape (no manual backward).

    Args:
        num_experts: number of experts.
        top_k: experts activated per token.
        expert_units: hidden size for 'relu' experts.
        expert_type: 'relu' (Dense-relu-Dense) or 'swiglu' (SwiGLUMLP).
        expert_hidden_dim: hidden dim for swiglu experts (default 2*dim).
        capacity_factor: kept for API compat / logging (no hard drop in
            dense evaluation).
        aux_loss_coef: scale for load-balancing aux loss (see aux_loss()).
    """

    def __init__(self, num_experts, top_k, expert_units=None,
                 expert_type='relu', expert_hidden_dim=None,
                 capacity_factor=1.0, aux_loss_coef=0.01, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_units = expert_units
        self.expert_type = expert_type
        self.expert_hidden_dim = expert_hidden_dim
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.experts = []
        self.aux_loss_value = 0.0

    def _build_expert(self, input_shape, hidden):
        if self.expert_type == 'swiglu':
            from .swiglu_mlp import SwiGLUMLP
            dim = input_shape[-1]
            return SwiGLUMLP(dim, hidden)
        e1 = Dense(hidden if hidden else self.expert_units, activation='relu')
        return e1

    def build(self, input_shape):
        from .swiglu_mlp import SwiGLUMLP
        self.input_dim = input_shape[-1]
        rng = np.random.default_rng(0)
        self.params['router_weight'] = Tensor(
            rng.normal(0, 0.02, (self.input_dim, self.num_experts)))
        hidden = self.expert_hidden_dim or self.expert_units or (self.input_dim * 2)
        self.experts = []
        for _ in range(self.num_experts):
            if self.expert_type == 'swiglu':
                ex = SwiGLUMLP(self.input_dim, hidden)
                ex.build(input_shape)
                self.experts.append(ex)
            else:
                e1 = Dense(hidden, activation='relu')
                e1.build(input_shape)
                mid = tuple(list(input_shape[:-1]) + [hidden])
                e2 = Dense(self.input_dim)
                e2.build(mid)
                self.experts.append([e1, e2])
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _expert_forward(self, expert, x, training):
        if isinstance(expert, list):
            out = x
            for layer in expert:
                out = layer(out, training=training)
            return out
        return expert(x, training=training)

    def forward(self, x, training=False):
        # x: Tensor (..., D); flatten tokens.
        x_shape = x.shape
        x_flat = x.reshape((-1, self.input_dim))
        logits = x_flat @ self.params['router_weight']  # (T, E) Tensor
        probs = autograd_ops.softmax(logits, axis=-1)  # Tensor
        probs_np = np.asarray(probs.data)

        # Top-k mask (non-differentiable indices, differentiable weights).
        top_k_idx = np.argsort(probs_np, axis=-1)[:, -self.top_k:]
        mask = np.zeros_like(probs_np)
        np.put_along_axis(mask, top_k_idx, 1.0, axis=-1)
        masked = probs * Tensor(mask)  # re-normalize over top-k
        renorm = masked.sum(axis=-1, keepdims=True)
        # Avoid div-by-zero: renorm is >= top-k min prob > 0 in practice;
        # add epsilon via numpy-safe Tensor op.
        weights = masked / (renorm + 1e-9)

        # Load-balancing aux stats (floats for logging/loss).
        tokens = probs_np.shape[0]
        frac = mask.sum(axis=0) / max(tokens, 1)
        mean_prob = probs_np.mean(axis=0)
        self.aux_loss_value = float(
            self.num_experts * np.sum(frac * mean_prob))
        self._last_frac = frac
        self._last_mean_prob = mean_prob

        out = None
        for e in range(self.num_experts):
            if np.all(mask[:, e] == 0):
                continue
            ex_out = self._expert_forward(self.experts[e], x_flat, training)
            # Keep Tensor path for router grad: slice differentiable weights.
            col = weights[:, e].reshape((-1, 1))
            scaled = ex_out * col
            out = scaled if out is None else out + scaled
        if out is None:
            out = x_flat * Tensor(np.zeros((1, 1)))
        return out.reshape(x_shape)

    def aux_loss(self):
        """Load-balancing loss (Switch-Transformer style)."""
        return self.aux_loss_coef * self.aux_loss_value
