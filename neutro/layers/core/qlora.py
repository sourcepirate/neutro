"""QLoRA — Quantized LoRA (Dettmers et al., 2023).

Stores the frozen base weight in 4-bit NormalFloat (NF4) with block-wise
absmax.  On every forward pass the weight is dequantized on-the-fly (to
float) and the LoRA residual is added:

    y = x @ dequant(W_nf4) + b + s * (x @ A) @ B

Only A and B are trainable — same as LoRA — but the base weight uses
~0.5 bytes/param instead of 8 bytes/param.

Double quantization: the per-block absmax scales are themselves quantized
to 8-bit per super-block when ``double_quant=True``.
"""
import numpy as np

from ..base import Layer
from ...initializers import get as get_initializer
from ...activations import get as get_activation
from neutro.autograd import Tensor
from neutro.utils.quantization import (
    quantize_nf4,
    dequantize_nf4,
    quantize_absmax,
    dequantize_absmax,
)


def _to_numpy(x):
    if isinstance(x, Tensor):
        return np.asarray(x.data)
    return np.asarray(x)


def _get_data(p):
    return p.data if isinstance(p, Tensor) else np.asarray(p)


class QLoRADense(Layer):
    """Quantized LoRA Dense layer.

    Args:
        units: output dim.
        rank: LoRA rank.
        alpha: LoRA alpha (scale = alpha / rank).
        dropout: dropout on LoRA branch input.
        use_bias: include (frozen) bias.
        activation: activation after the sum.
        block_size: NF4 block size (elements per absmax).
        double_quant: if True, quantize absmax scales to 8-bit.
        double_quant_block_size: super-block size for double quant.
        kernel_initializer: init for base weight before quantizing.
        bias_initializer: init for frozen bias.
        lora_initializer: init for A.
    """

    def __init__(
        self,
        units: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        use_bias: bool = True,
        activation=None,
        block_size: int = 64,
        double_quant: bool = True,
        double_quant_block_size: int = 256,
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
        self.block_size = int(block_size)
        self.double_quant = bool(double_quant)
        self.double_quant_block_size = int(double_quant_block_size)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.lora_initializer = get_initializer(lora_initializer)

        # Quantized storage (set in build).
        self.W_codes = None       # uint8 NF4 indices, shape (D, U)
        self.W_absmax = None      # float per-block absmax, or uint8 if double_quant
        self.W_absmax_dq = None   # dequantized absmax (when double_quant)
        self.W_scales2 = None     # second-level scales (when double_quant)
        self.W_shape = None
        self.b_frozen = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, input_shape) -> None:
        self.input_dim = input_shape[-1]

        W = self.kernel_initializer((self.input_dim, self.units)).astype(float)
        self.W_shape = W.shape
        codes, absmax, _ = quantize_nf4(W, block_size=self.block_size)
        self.W_codes = codes

        if self.double_quant:
            q_absmax, dq_absmax, scales2, _ = quantize_absmax(
                absmax, block_size=self.double_quant_block_size
            )
            self.W_absmax = q_absmax          # uint8
            self.W_absmax_dq = dq_absmax      # float (dequantized, used in forward)
            self.W_scales2 = scales2
            self._absmax_shape = absmax.shape
        else:
            self.W_absmax = absmax  # float
            self.W_absmax_dq = None
            self.W_scales2 = None

        if self.use_bias:
            self.b_frozen = self.bias_initializer((self.units,)).astype(float)
        else:
            self.b_frozen = None

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
    # Dequant helper
    # ------------------------------------------------------------------
    def _dequant_weight(self):
        """Dequantize W_codes -> float matrix (D, U)."""
        if self.double_quant:
            # First dequantize absmax.
            absmax = dequantize_absmax(
                self.W_absmax, self.W_scales2,
                block_size=self.double_quant_block_size,
                shape=self._absmax_shape,
            )
        else:
            absmax = self.W_absmax
        W = dequantize_nf4(self.W_codes, absmax, block_size=self.block_size, shape=self.W_shape)
        return W

    def get_merged_weight(self):
        """Return dequant(W) + s*A@B without mutating state."""
        W_dq = self._dequant_weight()
        delta = self.scaling * (_get_data(self.params["lora_A"]) @ _get_data(self.params["lora_B"]))
        return W_dq + delta

    @classmethod
    def from_dense(cls, dense_layer, rank=8, alpha=16, dropout=0.0, block_size=64, double_quant=True, **kwargs):
        """Create QLoRADense from an existing Dense layer (quantizes its W)."""
        if not getattr(dense_layer, "built", False):
            raise ValueError("from_dense requires a built Dense layer")
        name = kwargs.pop("name", (dense_layer.name + "_qlora") if dense_layer.name else None)
        use_bias = kwargs.pop("use_bias", dense_layer.use_bias)
        activation = kwargs.pop("activation", dense_layer.activation_name)
        obj = cls(
            units=dense_layer.units,
            rank=rank, alpha=alpha, dropout=dropout,
            use_bias=use_bias, activation=activation,
            block_size=block_size, double_quant=double_quant,
            name=name, **kwargs,
        )
        obj.input_dim = dense_layer.input_dim
        W_data = _get_data(dense_layer.params["W"]).astype(float)
        obj.W_shape = W_data.shape
        codes, absmax, _ = quantize_nf4(W_data, block_size=obj.block_size)
        obj.W_codes = codes
        if obj.double_quant:
            q_absmax, dq_absmax, scales2, _ = quantize_absmax(absmax, block_size=obj.double_quant_block_size)
            obj.W_absmax = q_absmax
            obj.W_absmax_dq = dq_absmax
            obj.W_scales2 = scales2
            obj._absmax_shape = absmax.shape
        else:
            obj.W_absmax = absmax
            obj.W_absmax_dq = None
            obj.W_scales2 = None
        if use_bias and "b" in dense_layer.params:
            obj.b_frozen = _get_data(dense_layer.params["b"]).astype(float).copy()
        else:
            obj.b_frozen = None
        obj.params["lora_A"] = Tensor(obj.lora_initializer((obj.input_dim, obj.rank)).astype(float))
        obj.params["lora_B"] = Tensor(np.zeros((obj.rank, obj.units), dtype=float))
        obj.built = True
        obj.input_shape = dense_layer.input_shape
        return obj

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs, training=False):
        if isinstance(inputs, Tensor):
            x = inputs.data
        else:
            x = np.asarray(inputs, dtype=float)

        self._cache_x = x
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)  # (N, D)

        # Dropout on LoRA branch input.
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*x_flat.shape) >= self.dropout_rate).astype(float)
            mask /= (1.0 - self.dropout_rate)
            self._cache_dropout_mask = mask
            x_lora = x_flat * mask
        else:
            self._cache_dropout_mask = None
            x_lora = x_flat

        # Dequantize weight on the fly.
        W_dq = self._dequant_weight()  # (D, U)
        self._cache_W_dq = W_dq  # 🔍 for dx via W in backward

        A = _get_data(self.params["lora_A"])  # (D, r)
        B = _get_data(self.params["lora_B"])  # (r, U)
        self._cache_A = A
        self._cache_B = B
        self._cache_x_lora = x_lora
        h = x_lora @ A  # (N, r)
        self._cache_h = h

        lora_out = h @ B * self.scaling  # (N, U)
        base_out = x_flat @ W_dq  # (N, U)

        z_flat = base_out + lora_out
        if self.b_frozen is not None:
            z_flat = z_flat + self.b_frozen

        self._cache_z_pre = z_flat.reshape(*orig_shape[:-1], self.units)

        z = z_flat.reshape(*orig_shape[:-1], self.units)

        if self.activation is not None:
            out = self.activation(z)
            if isinstance(out, Tensor):
                out = out.data
            return Tensor(out) if isinstance(inputs, Tensor) else np.asarray(out)

        if isinstance(inputs, Tensor):
            return Tensor(z)
        return z

    # ------------------------------------------------------------------
    # Backward — same math as LoRADense, but dx via W uses W_dq.
    # ------------------------------------------------------------------
    def backward(self, grad_output):
        g = _to_numpy(grad_output)
        g_flat = g.reshape(-1, self.units)

        if self.activation is not None:
            z_flat = self._cache_z_pre.reshape(-1, self.units)
            if hasattr(self.activation, "gradient_fast"):
                g_flat = self.activation.gradient_fast(z_flat, g_flat)
            else:
                g_flat = g_flat * self.activation.gradient(z_flat)

        x = self._cache_x
        x_lora = self._cache_x_lora
        h = self._cache_h
        A = _get_data(self.params["lora_A"])
        B = _get_data(self.params["lora_B"])
        W_dq = self._cache_W_dq

        # Grads for LoRA params.
        self.grads["lora_B"] = self.scaling * (h.T @ g_flat)  # (r, U)
        dh = self.scaling * (g_flat @ B.T)                      # (N, r)
        self.grads["lora_A"] = x_lora.T @ dh                    # (D, r)

        # Grad w.r.t. inputs.
        dx_W = g_flat @ W_dq.T
        dx_lora = dh @ A.T
        if self._cache_dropout_mask is not None:
            dx_lora = dx_lora * self._cache_dropout_mask

        dx_flat = dx_W + dx_lora
        dx = dx_flat.reshape(x.shape)
        return dx

    def count_params(self):
        return super().count_params()

    def memory_bytes(self):
        """Estimate memory for frozen weight: NF4 ~0.5 bytes/param + absmax overhead."""
        n = self.W_codes.size
        # 4-bit = 0.5 bytes per param
        base = n * 0.5
        # absmax overhead: one float per block, or 1 byte if double-quantized
        num_blocks = self.W_absmax.size if hasattr(self.W_absmax, "size") else 0
        if self.double_quant:
            overhead = num_blocks * 1.0  # int8
            overhead += self.W_scales2.size * 4  # float32 second-level
        else:
            overhead = num_blocks * 4  # float
        lora = self.params["lora_A"].data.size * 8 + self.params["lora_B"].data.size * 8
        return {"base_nf4": base, "absmax_overhead": overhead, "lora_fp64": lora,
                "total": base + overhead + lora}


# ---------------------------------------------------------------------------
# apply_qlora — inject QLoRA into an existing model
# ---------------------------------------------------------------------------

def _replace_layer_q(parent, old_layer, new_layer):
    """Replace old_layer with new_layer inside parent (same as LoRA helper)."""
    from ..base import Layer as _Layer
    for attr_name, attr_val in list(vars(parent).items()):
        if attr_val is old_layer:
            setattr(parent, attr_name, new_layer)
            return True
        if isinstance(attr_val, list):
            for i, item in enumerate(attr_val):
                if item is old_layer:
                    attr_val[i] = new_layer
                    return True
                if isinstance(item, _Layer) and _replace_layer_q(item, old_layer, new_layer):
                    return True
        elif isinstance(attr_val, tuple):
            lst = list(attr_val)
            replaced = False
            for i, item in enumerate(lst):
                if item is old_layer:
                    lst[i] = new_layer
                    replaced = True
                elif isinstance(item, _Layer) and _replace_layer_q(item, old_layer, new_layer):
                    replaced = True
            if replaced:
                setattr(parent, attr_name, tuple(lst))
                return True
        elif isinstance(attr_val, dict):
            for k, v in list(attr_val.items()):
                if v is old_layer:
                    attr_val[k] = new_layer
                    return True
                if isinstance(v, _Layer) and _replace_layer_q(v, old_layer, new_layer):
                    return True
        elif isinstance(attr_val, _Layer):
            if _replace_layer_q(attr_val, old_layer, new_layer):
                return True
    return False


def apply_qlora(model, rank=8, alpha=16, dropout=0.0, block_size=64,
                double_quant=True, target_modules=None, verbose=True):
    """Inject QLoRA into Dense layers of a (built) model.

    Same interface as ``apply_lora`` but quantizes the frozen weight to NF4.

    Args:
        model: a built Sequential or Model.
        rank, alpha, dropout: LoRA hyperparams.
        block_size: NF4 block size.
        double_quant: whether to double-quantize absmax scales.
        target_modules: None (all Dense) or name substrings / explicit layers.
        verbose: print summary.

    Returns:
        (model, list[QLoRADense])
    """
    from .dense import Dense
    from ..base import Layer as _Layer

    all_layers = model._get_all_layers()

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
    elif isinstance(target_modules, list) and target_modules and isinstance(target_modules[0], _Layer):
        targets = [l for l in target_modules if isinstance(l, Dense)]
    elif isinstance(target_modules, list):
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
            print("[apply_qlora] No matching Dense layers found — nothing to adapt.")
        return model, []

    new_layers = []
    for dense in targets:
        qlora = QLoRADense.from_dense(
            dense, rank=rank, alpha=alpha, dropout=dropout,
            block_size=block_size, double_quant=double_quant,
        )
        replaced = False
        if hasattr(model, "layers") and dense in model.layers:
            idx = model.layers.index(dense)
            model.layers[idx] = qlora
            replaced = True
        else:
            if _replace_layer_q(model, dense, qlora):
                replaced = True
            else:
                for child in list(getattr(model, "layers", [])):
                    if isinstance(child, _Layer) and _replace_layer_q(child, dense, qlora):
                        replaced = True
                        break
        if not replaced and verbose:
            print(f"[apply_qlora] Warning: could not locate parent of '{dense.name}' — skipping.")
            continue
        new_layers.append(qlora)
        if verbose:
            print(f"[apply_qlora] Adapted '{dense.name or dense.__class__.__name__}' "
                  f"({dense.input_dim} -> {dense.units}) with rank={rank}, alpha={alpha}, "
                  f"block_size={block_size}, double_quant={double_quant}")

    if verbose:
        trainable = sum(l.count_params() for l in new_layers)
        frozen = sum(l.W_codes.size for l in new_layers)
        print(f"[apply_qlora] Total: {len(new_layers)} layers adapted | "
              f"trainable {trainable:,} params vs {frozen:,} frozen "
              f"({100*trainable/max(frozen,1):.2f}% of base)")

    return model, new_layers
