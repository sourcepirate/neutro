import numpy as np

from ..base import Layer
from .dense import Dense
from neutro.autograd import Tensor, ops as autograd_ops


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
            expert = SwiGLUMLP(self.input_dim, hidden)
            expert.build(input_shape)
            return expert

        first = Dense(hidden, activation='relu')
        first.build(input_shape)
        second = Dense(self.input_dim)
        second.build((*input_shape[:-1], hidden))
        return [first, second]

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        rng = np.random.default_rng(0)
        self.params['router_weight'] = Tensor(
            rng.normal(0, 0.02, (self.input_dim, self.num_experts)))
        hidden = self.expert_hidden_dim or self.expert_units or (self.input_dim * 2)
        self.experts = [self._build_expert(input_shape, hidden) for _ in range(self.num_experts)]
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _expert_forward(self, expert, x, training):
        if not isinstance(expert, list):
            return expert(x, training=training)
        for layer in expert:
            x = layer(x, training=training)
        return x

    def _top_k_mask(self, probs):
        top_k_idx = np.argsort(probs, axis=-1)[:, -self.top_k:]
        mask = np.zeros_like(probs)
        np.put_along_axis(mask, top_k_idx, 1.0, axis=-1)
        return mask

    def _update_aux_stats(self, mask, probs):
        tokens = probs.shape[0]
        self._last_frac = mask.sum(axis=0) / max(tokens, 1)
        self._last_mean_prob = probs.mean(axis=0)
        self.aux_loss_value = float(
            self.num_experts * np.sum(self._last_frac * self._last_mean_prob))

    def forward(self, x, training=False):
        x_shape = x.shape
        x_flat = x.reshape((-1, self.input_dim))
        probs = autograd_ops.softmax(x_flat @ self.params['router_weight'], axis=-1)
        probs_np = np.asarray(probs.data)

        mask = self._top_k_mask(probs_np)
        masked = probs * Tensor(mask)
        weights = masked / (masked.sum(axis=-1, keepdims=True) + 1e-9)
        self._update_aux_stats(mask, probs_np)

        out = None
        for index, expert in enumerate(self.experts):
            if not np.any(mask[:, index]):
                continue
            weight = weights[:, index].reshape((-1, 1))
            scaled = self._expert_forward(expert, x_flat, training) * weight
            out = scaled if out is None else out + scaled

        if out is None:
            out = x_flat * Tensor(np.zeros((1, 1)))
        return out.reshape(x_shape)

    def aux_loss(self):
        """Load-balancing loss (Switch-Transformer style)."""
        return self.aux_loss_coef * self.aux_loss_value
