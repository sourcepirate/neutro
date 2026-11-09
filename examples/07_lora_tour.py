"""A tour of LoRA: freeze the base weight, train a low-rank update.

Run from the repository root::

    python examples/07_lora_tour.py

It compares a Dense layer with LoRADense, shows the parameter savings, checks
that the initial output is unchanged, and merges the adapter back into the base
weight.
"""

import numpy as np

from neutro.layers import Dense, LoRADense


def main():
    rng = np.random.default_rng(0)

    input_dim, units, rank = 64, 32, 4
    x = rng.normal(size=(2, input_dim))

    base = Dense(units)
    base.build((2, input_dim))

    lora = LoRADense(units, rank=rank, alpha=8)
    lora.build((2, input_dim))

    # Start from the same frozen base weight.
    lora.W_frozen = base.params["W"].data.copy()
    lora.b_frozen = base.params["b"].data.copy()

    print("Parameter counts")
    print("  Dense params:        ", base.count_params())
    print("  LoRA trainable params:", lora.count_params())
    print(
        "  reduction:           ",
        f"{100 * (1 - lora.count_params() / base.count_params()):.1f}%",
    )
    print("  trainable keys:      ", sorted(lora.params))
    print()

    print("Initial output matches the frozen base:")
    print(" ", np.allclose(base(x).data, lora(x).data))
    print()

    # Give the low-rank branch a non-zero update.
    lora.params["lora_B"].data = rng.normal(size=(rank, units)) * 0.05
    before_merge = lora(x).data.copy()

    lora.merge_weights()
    after_merge = lora(x).data

    print("After merge_weights()")
    print("  output unchanged:    ", np.allclose(before_merge, after_merge))
    print("  merged W shape:      ", lora.W_frozen.shape)
    print("  lora_B zeroed:       ", np.allclose(lora.params["lora_B"].data, 0.0))
    print()

    # Manual backward still fills only the LoRA gradients.
    upstream = rng.normal(size=(2, units))
    dx = lora.backward(upstream)
    print("Manual backward")
    print("  grad shapes:         ", {k: v.shape for k, v in lora.grads.items()})
    print("  input grad shape:    ", dx.shape)
    print()
    print("Only lora_A and lora_B are in params; W_frozen is not trainable.")
    print("Next: docs/tutorials/07-lora-finetuning.md")


if __name__ == "__main__":
    main()
