# Tutorial 04 — Build a Tiny Transformer LM

**Time:** about 20 minutes. **Prerequisites:** [Tutorial 03](03-attention-from-scratch.md).

Now we put the pieces together: embeddings, a causal transformer block, a
language-model head, training, and autoregressive generation.

Run the companion script:

```bash
python examples/04_tiny_transformer_lm.py
```

Expected output, approximately:

```text
Training loss (first, middle, last):
  1.3523 0.0002 0.0001

Input:      [13 10  8  4  4  0  1  0  2 13]
Target:     [14 11  9  5  5  1  2  1  3 14]
Prediction: [14 11  9  5  5  1  2  1  3 14]

Prompt:    [[13 10  8  4]]
Generated: [[13 10  8  4  5  6  7  8  9]]
```

The task is synthetic and exact: the target is always `(token + 1) % vocab`.
If the model learns it, predictions match targets and generation continues the
incrementing sequence.

---

## 1. Data

```python
VOCAB_SIZE = 16
SEQ_LEN = 12

x = rng.integers(0, VOCAB_SIZE, size=(256, SEQ_LEN))
y = (x + 1) % VOCAB_SIZE
```

```text
x: (B, S)     integer token ids
y: (B, S)     the next token at every position
```

This is a **causal** language-model task: position `t` predicts position `t+1`
using only positions `0..t`.

---

## 2. The model

```python
model = Sequential([
    Embedding(VOCAB_SIZE, 32, input_shape=(SEQ_LEN,)),
    TransformerBlock(
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        causal=True,
        pre_norm=True,
    ),
    Dense(VOCAB_SIZE),   # raw logits
])
```

Shape walkthrough:

```text
tokens:       (B, S)              integer ids
Embedding:    (B, S, 32)          token vectors
Transformer:  (B, S, 32)          same shape
Dense:        (B, S, 16)          one logit per vocabulary item
```

The final layer has **no softmax**. It returns logits. The loss applies
softmax internally, and `generate()` applies it when sampling.

---

## 3. Loss and training

```python
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss=SparseCategoricalCrossentropy(from_logits=True),
)

history = model.fit(x, y, epochs=10, batch_size=32, verbose=0)
```

`SparseCategoricalCrossentropy(from_logits=True)` expects:

```text
y_true: (B, S)       integer ids
y_pred: (B, S, V)    raw logits
```

It uses the numerically stable combined softmax + cross-entropy gradient. The
loss should fall quickly on this easy task.

---

## 4. Check the predictions

```python
logits = model.predict(x[:2])
predictions = np.argmax(logits.data, axis=-1)

print(x[0][:10])
print(y[0][:10])
print(predictions[0][:10])
```

Expected: the prediction equals the target, i.e. each token is the previous
token plus one.

`predict` returns a `Tensor`; `.data` gives the NumPy array.

---

## 5. Generate

```python
prompt = x[:1, :4]
generated = model.generate(prompt, max_new_tokens=5)
print(generated)
```

`Model.generate`:

1. Runs the prompt through the model.
2. Takes the logits at the last position.
3. Divides by `temperature`, applies softmax, and samples one token.
4. Appends it and repeats, using a KV cache so past keys/values are reused.

Shape:

```text
prompt:    (1, 4)
generated: (1, 9)    prompt + 5 new tokens
```

For this task, the model should continue `13, 10, 8, 4` with `5, 6, 7, 8, 9`.

---

## 6. What the transformer block is doing

```text
x
├── LayerNorm -> Attention(causal) -> Dropout -> + x     (residual)
└── LayerNorm -> Dense(expand) -> ReLU -> Dense(project)
    -> Dropout -> + previous                            (residual)
```

- The causal mask prevents position `t` from seeing the future.
- The residual connections keep gradients flowing.
- Pre-norm (`pre_norm=True`) normalizes before each sublayer, which is the
  modern GPT/Llama ordering.

See [TransformerBlock](../layers/transformer/transformer_block.md) for the
line-by-line explanation.

---

## 7. Your turn

1. Change `causal=False`. What happens to the loss? Why can the model cheat?
2. Remove the `TransformerBlock`. Can a single `Dense` layer learn the task?
3. Increase `SEQ_LEN` to 32. Does the model still learn?
4. Change the task to `(token * 2) % vocab`. Is it harder? Why?
5. Set `temperature=0.1` in `generate`. Is the output more or less random?

<details>
<summary>Answers</summary>

1. Without causality, position `t` can see `t+1` directly, so the loss can drop
   without learning a real next-token rule.
2. A single Dense layer can learn a position-independent mapping if the task is
   simple; the transformer matters when relationships depend on context.
3. Usually yes, but it needs more training. Longer sequences also make attention
   more expensive.
4. It depends on whether the mapping is still one-to-one. Multiplication can
   collide for some vocab sizes, making the task harder.
5. Lower temperature makes sampling more greedy and less random.

</details>

---

## What to read next

- [TransformerBlock](../layers/transformer/transformer_block.md)
- [Multi-Head Attention](../layers/attention/mha.md)
- [KV cache](../layers/attention/kv_cache.md)
- [Tutorial 05 — Manual backprop (advanced)](05-manual-backprop.md) *(next)*
