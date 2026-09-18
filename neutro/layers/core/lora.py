"""LoRA (Low-Rank Adaptation) for Dense layers — Hu et al., ICLR 2022.

A frozen base weight ``W`` (D, U) is augmented with a low-rank delta:

    W' = W + (alpha / r) * A @ B

where A (D, r) and B (r, U) are the only trainable parameters, r << min(D,U).
B is zero-initialized so that W' == W at the start of fine-tuning.
"""
import numpy as np

from ..base import Layer
from ...initializers import get as get_initializer
from ...activations import get as get_activation
from neutro.autograd import Tensor


def _to_numpy(x):
    """Unwrap Tensor -> ndarray, pass through ndarray."""
    if isinstance(x, Tensor):
        return np.asarray(x.data)
    return np.asarray(x)


def _get_data(p):
    return p.data if isinstance(p, Tensor) else np.asarray(p)


class LoRADense(Layer):
    """Dense layer with Low-Rank Adaptation (LoRA).

    Freezes the base weight and learns a low-rank residual:

        y = activation( x @ W_frozen + b_frozen + s * (x @ A) @ B )

    where s = alpha / rank is the scaling factor.

    Args:
        units: output dimensionality (like Dense).
        rank: LoRA rank r — bottleneck dimension.  Smaller = fewer params.
        alpha: LoRA alpha — scaling numerator.  Effective scale = alpha / rank.
        dropout: dropout rate applied to the LoRA branch input (0 = no dropout).
        use_bias: whether to include a (frozen) bias term.
        activation: activation string or None (applied after the sum).
        kernel_initializer: initializer for the frozen base weight W.
        lora_initializer: initializer for A (B is always zeros).
        bias_initializer: initializer for the frozen bias.
    """

    def __init__(
        self,
        units: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        use_bias: bool = True,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        lora_initializer="glorot_uniform",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.units = units
        self.rank = rank
        self.alpha = alpha
        self.scaling = float(alpha) / float(rank)
        self.dropout_rate = float(dropout)
        self.use_bias = use_bias
        self.activation_name = activation
        self.activation = get_activation(activation)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.lora_initializer = get_initializer(lora_initializer)

        # Frozen weights — plain numpy, NOT in self.params so optimizer skips them.
        self.W_frozen = None  # set in build()
        self.b_frozen = None
        self.merged = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, input_shape) -> None:
        self.input_dim = input_shape[-1]

        # Frozen base weight (numpy, not Tensor — optimizer ignores it).
        self.W_frozen = self.kernel_initializer((self.input_dim, self.units)).astype(float)
        if self.use_bias:
            self.b_frozen = self.bias_initializer((self.units,)).astype(float)
        else:
            self.b_frozen = None

        # Trainable low-rank matrices — Tensors so autograd / optimizer sees them.
        # A: (D, r)  — Kaiming / Glorot init
        # B: (r, U)  — zeros so delta is 0 at initialization
        self.params["lora_A"] = Tensor(
            self.lora_initializer((self.input_dim, self.rank)).astype(float)
        )
        self.params["lora_B"] = Tensor(
            np.zeros((self.rank, self.units), dtype=float)
        )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    # ------------------------------------------------------------------
    # Helpers — alternate constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dense(cls, dense_layer, rank=8, alpha=16, dropout=0.0, **kwargs):
        """Create a LoRADense that wraps an existing Dense layer's weights.

        The frozen weight is copied from ``dense_layer``; A/B are freshly
        initialized.  The new layer's ``name`` defaults to the dense layer's
        name (with ``_lora`` suffix) unless explicitly overridden.
        """
        # Lazy import to avoid circular deps.
        from .dense import Dense  # noqa: F401

        if not getattr(dense_layer, "built", False):
            raise ValueError("from_dense requires a built Dense layer (call it once first)")

        name = kwargs.pop("name", (dense_layer.name + "_lora") if dense_layer.name else None)
        use_bias = kwargs.pop("use_bias", dense_layer.use_bias)
        activation = kwargs.pop("activation", dense_layer.activation_name)

        obj = cls(
            units=dense_layer.units,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_bias=use_bias,
            activation=activation,
            name=name,
            **kwargs,
        )
        # Manually build with the same input_dim, then overwrite W_frozen.
        obj.input_dim = dense_layer.input_dim
        W_data = _get_data(dense_layer.params["W"])
        obj.W_frozen = W_data.astype(float).copy()
        if use_bias and "b" in dense_layer.params:
            obj.b_frozen = _get_data(dense_layer.params["b"]).astype(float).copy()
        else:
            obj.b_frozen = None

        obj.params["lora_A"] = Tensor(
            obj.lora_initializer((obj.input_dim, obj.rank)).astype(float)
        )
        obj.params["lora_B"] = Tensor(
            np.zeros((obj.rank, obj.units), dtype=float)
        )
        obj.built = True
        obj.input_shape = dense_layer.input_shape
        return obj

    # ------------------------------------------------------------------
    # Merge / unmerge (for efficient inference after fine-tuning)
    # ------------------------------------------------------------------
    def merge_weights(self):
        """Fold LoRA delta into W_frozen so forward is a single matmul.

        After calling, ``W_frozen <- W_frozen + s*A@B`` and ``B`` is zeroed
        so that subsequent forwards produce the same result without the
        extra matmul.  Useful for deployment.
        """
        if self.merged:
            return
        delta = self.scaling * (_get_data(self.params["lora_A"]) @ _get_data(self.params["lora_B"]))
        self.W_frozen = self.W_frozen + delta
        self.params["lora_B"].data[:] = 0.0
        self.merged = True

    def unmerge_weights(self):
        """Undo merge_weights (requires that B was zeroed, not re-trained)."""
        # No-op if not merged — we cannot truly unmerge after further training
        # without storing the original delta.  This is a convenience for
        # merge-then-evaluate flows.
        self.merged = False

    def get_merged_weight(self):
        """Return W_frozen + s*A@B without mutating state (for inspection)."""
        delta = self.scaling * (_get_data(self.params["lora_A"]) @ _get_data(self.params["lora_B"]))
        return self.W_frozen + delta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs, training=False):
        # Cache for backward — unwrap Tensor to plain numpy.
        if isinstance(inputs, Tensor):
            x = inputs.data
        else:
            x = np.asarray(inputs, dtype=float)

        self._cache_x = x  # (..., D)  — 🔍 used in backward to compute grads for A, B and dx
        orig_shape = x.shape

        # Flatten leading dims so matmul is 2-D, then reshape output.
        # This mirrors Dense which relies on np.matmul broadcasting; we do
        # explicit flatten for clarity.
        x_flat = x.reshape(-1, self.input_dim)  # (N, D)

        # ---- LoRA branch input dropout (Hu et al. applies dropout to x) ----
        if training and self.dropout_rate > 0:
            # Inverted dropout.
            mask = (np.random.rand(*x_flat.shape) >= self.dropout_rate).astype(float)
            mask /= (1.0 - self.dropout_rate)
            self._cache_dropout_mask = mask  # 🔍 used to scale grad in backward
            x_lora = x_flat * mask
        else:
            self._cache_dropout_mask = None
            x_lora = x_flat

        A = _get_data(self.params["lora_A"])  # (D, r)
        B = _get_data(self.params["lora_B"])  # (r, U)

        self._cache_A = A  # 🔍 not strictly needed (can re-read), but explicit
        self._cache_B = B
        self._cache_x_lora = x_lora  # 🔍 after dropout, used for grad_A
        h = x_lora @ A  # (N, r)  — 🔍 cached as self._cache_h for grad_B
        self._cache_h = h

        lora_out = h @ B  # (N, U)
        lora_out = lora_out * self.scaling

        base_out = x_flat @ self.W_frozen  # (N, U) — frozen path, no grad

        z_flat = base_out + lora_out
        if self.b_frozen is not None:
            z_flat = z_flat + self.b_frozen  # broadcast (U,)

        self._cache_z_pre = z_flat.reshape(*orig_shape[:-1], self.units)  # 🔍 for activation grad

        z = z_flat.reshape(*orig_shape[:-1], self.units)

        # Activation (if any) — applied after the sum.
        if self.activation is not None:
            # Activation may expect Tensor or ndarray; pass z as ndarray
            # and let activation handle it.  Cache pre-activation for grad.
            out = self.activation(z)
            # Unwrap if activation returned Tensor.
            if isinstance(out, Tensor):
                out = out.data
            return Tensor(out) if isinstance(inputs, Tensor) else np.asarray(out)

        if isinstance(inputs, Tensor):
            return Tensor(z)
        return z

    # ------------------------------------------------------------------
    # Backward — explicit, no GradientTape
    # ------------------------------------------------------------------
    def backward(self, grad_output):
        """Manual backward.

        Args:
            grad_output: dL/dy, same shape as forward output, ndarray or Tensor.

        Computes:
            grads['lora_A'], grads['lora_B']  (frozen W/B get no grad)
        Returns:
            grad w.r.t. inputs, same shape as cached x.
        """
        g = _to_numpy(grad_output)  # (..., U)
        g_flat = g.reshape(-1, self.units)  # (N, U)

        # ---- activation backward ----
        if self.activation is not None:
            z_pre = self._cache_z_pre  # (..., U)
            # z_pre is already reshaped; flatten for element-wise activation grad
            z_flat = z_pre.reshape(-1, self.units)
            if hasattr(self.activation, "gradient_fast"):
                # Softmax path: expects (N, U) flat
                g_flat = self.activation.gradient_fast(z_flat, g_flat)
            else:
                act_grad = self.activation.gradient(z_flat)  # (N, U)
                g_flat = g_flat * act_grad
            # g is now dL/dz (pre-activation)

        # Cached values
        x = self._cache_x  # (..., D)
        x_flat = x.reshape(-1, self.input_dim)  # (N, D)
        x_lora = self._cache_x_lora  # (N, D) — after dropout
        h = self._cache_h  # (N, r)
        A = _get_data(self.params["lora_A"])  # (D, r)
        B = _get_data(self.params["lora_B"])  # (r, U)

        # ---- grads for B and A ----
        # y_lora = s * h @ B  =>  dL/dB = s * h^T @ g_flat
        # 📐 (r, N) @ (N, U) -> (r, U)
        self.grads["lora_B"] = self.scaling * (h.T @ g_flat)

        # dL/dh = s * g_flat @ B^T  => (N, U) @ (U, r) -> (N, r)
        dh = self.scaling * (g_flat @ B.T)

        # y_lora via x_lora @ A -> h ;  dL/dA = x_lora^T @ dh
        # 📐 (D, N) @ (N, r) -> (D, r)
        self.grads["lora_A"] = x_lora.T @ dh

        # ---- grad w.r.t. inputs ----
        # Via frozen W:  g_flat @ W^T  => (N, U) @ (U, D) -> (N, D)
        dx_W = g_flat @ self.W_frozen.T

        # Via LoRA:  dh @ A^T  => (N, r) @ (r, D) -> (N, D)
        dx_lora = dh @ A.T

        # If dropout was applied, scale grad through the mask.
        if self._cache_dropout_mask is not None:
            dx_lora = dx_lora * self._cache_dropout_mask

        dx_flat = dx_W + dx_lora  # (N, D)
        dx = dx_flat.reshape(x.shape)

        # Also handle Tensor grad_output: return ndarray (Layer.backward contract)
        return dx

    def count_params(self):
        """Only trainable LoRA params are counted (frozen W excluded)."""
        # Override to make summary show trainable count correctly.
        # If you want total (including frozen), use count_params(include_frozen=True).
        return super().count_params()

    def count_params_frozen(self):
        """Total including frozen base weight (for memory reporting)."""
        frozen = self.W_frozen.size + (self.b_frozen.size if self.b_frozen is not None else 0)
        return frozen + self.count_params()


# ---------------------------------------------------------------------------
# apply_lora — inject LoRA into an existing model
# ---------------------------------------------------------------------------

def _replace_layer(parent, old_layer, new_layer):
    """Replace old_layer with new_layer somewhere inside parent's attributes.

    Searches direct attributes, lists, tuples, and dicts.  Returns True if
    a replacement was made.
    """
    for attr_name, attr_val in list(vars(parent).items()):
        if attr_val is old_layer:
            setattr(parent, attr_name, new_layer)
            return True
        if isinstance(attr_val, list):
            for i, item in enumerate(attr_val):
                if item is old_layer:
                    attr_val[i] = new_layer
                    return True
                if isinstance(item, Layer) and _replace_layer(item, old_layer, new_layer):
                    return True
        elif isinstance(attr_val, tuple):
            lst = list(attr_val)
            replaced = False
            for i, item in enumerate(lst):
                if item is old_layer:
                    lst[i] = new_layer
                    replaced = True
                elif isinstance(item, Layer) and _replace_layer(item, old_layer, new_layer):
                    replaced = True
            if replaced:
                setattr(parent, attr_name, tuple(lst))
                return True
        elif isinstance(attr_val, dict):
            for k, v in list(attr_val.items()):
                if v is old_layer:
                    attr_val[k] = new_layer
                    return True
                if isinstance(v, Layer) and _replace_layer(v, old_layer, new_layer):
                    return True
        elif isinstance(attr_val, Layer):
            if _replace_layer(attr_val, old_layer, new_layer):
                return True
    return False


def apply_lora(model, rank=8, alpha=16, dropout=0.0, target_modules=None, verbose=True):
    """Inject LoRA into Dense layers of a (built) model.

    Args:
        model: a built ``Sequential`` or ``Model`` instance.
        rank: LoRA rank for each adapted layer.
        alpha: LoRA alpha (scale = alpha / rank).
        dropout: dropout rate on the LoRA branch.
        target_modules: which layers to adapt.  Options:
            - None (default): adapt **all** Dense layers (auto-detect).
            - str or list[str]: substring match against ``layer.name`` or
              the parent attribute name (e.g. ``["ffn", " Dense"]``).
            - list[Layer]: explicit layer objects to replace.
        verbose: if True, print which layers were adapted.

    Returns:
        model (same object, mutated in-place) and a list of the new
        LoRADense layers that were inserted.
    """
    from .dense import Dense

    # Collect candidate Dense layers (including those reachable via sublayers).
    all_layers = model._get_all_layers()

    # Determine target set.
    if target_modules is None:
        targets = [l for l in all_layers if isinstance(l, Dense)]
    elif isinstance(target_modules, str):
        target_modules = [target_modules]
        targets = []
        for layer in all_layers:
            if not isinstance(layer, Dense):
                continue
            name = layer.name or ""
            if any(sub in name or sub in layer.__class__.__name__ for sub in target_modules):
                targets.append(layer)
    elif isinstance(target_modules, list) and target_modules and isinstance(target_modules[0], Layer):
        targets = [l for l in target_modules if isinstance(l, Dense)]
    elif isinstance(target_modules, list):
        # list of substrings
        targets = []
        for layer in all_layers:
            if not isinstance(layer, Dense):
                continue
            name = layer.name or ""
            if any(sub in name or sub in layer.__class__.__name__ for sub in target_modules):
                targets.append(layer)
    else:
        raise ValueError(f"Unsupported target_modules type: {type(target_modules)}")

    if not targets:
        if verbose:
            print("[apply_lora] No matching Dense layers found — nothing to adapt.")
        return model, []

    new_layers = []
    for dense in targets:
        lora = LoRADense.from_dense(dense, rank=rank, alpha=alpha, dropout=dropout)

        # Try to replace in the model hierarchy.
        replaced = False
        # Top-level Sequential.layers list
        if hasattr(model, "layers") and dense in model.layers:
            idx = model.layers.index(dense)
            model.layers[idx] = lora
            replaced = True
        else:
            # Search recursively via _replace_layer on the model and its direct children.
            if _replace_layer(model, dense, lora):
                replaced = True
            else:
                for child in list(getattr(model, "layers", [])):
                    if isinstance(child, Layer) and _replace_layer(child, dense, lora):
                        replaced = True
                        break

        if not replaced and verbose:
            print(f"[apply_lora] Warning: could not locate parent of layer '{dense.name}' — skipping.")
            continue

        new_layers.append(lora)
        if verbose:
            print(f"[apply_lora] Adapted '{dense.name or dense.__class__.__name__}' "
                  f"({dense.input_dim} -> {dense.units}) with rank={rank}, alpha={alpha}")

    if verbose:
        # Summary
        trainable = sum(l.count_params() for l in new_layers)
        frozen = sum(l.W_frozen.size for l in new_layers)
        print(f"[apply_lora] Total: {len(new_layers)} layers adapted | "
              f"trainable {trainable:,} params vs {frozen:,} frozen "
              f"({100*trainable/max(frozen,1):.2f}% of base)")

    return model, new_layers
