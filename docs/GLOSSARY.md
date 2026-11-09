# Glossary

Short definitions for the words that show up everywhere. Each entry links to a
place where you can see the idea in code or in a longer explanation.

| Term | Plain-English meaning |
|------|-----------------------|
| **Tensor** | An array plus a gradient slot. In Neutro, `Tensor` wraps a NumPy array; it does not carry a `requires_grad` flag. See [`Tensor`](autograd/tensor.md). |
| **Parameter** | A learned array owned by a layer, stored in `layer.params` (for example `W` and `b`). |
| **Gradient** | The derivative of the loss with respect to a parameter. Stored in `layer.grads` (and on `Tensor.grad`). |
| **Forward pass** | Running inputs through the model to get predictions. |
| **Backward pass** | Propagating the loss gradient backward to fill parameter gradients. |
| **GradientTape** | A recorder that remembers differentiable operations so it can compute gradients later. See [`GradientTape`](autograd/tape.md). |
| **Layer** | A reusable computation with optional parameters, e.g. `Dense`, `Conv2D`, `LSTM`. |
| **Model** | A `Layer` that owns other layers and knows how to train. `Sequential` is the simplest model. |
| **Batch** | A group of samples processed together. Shape usually starts with `B`. |
| **Epoch** | One full pass over the training data. |
| **Logits** | Raw, unnormalized class scores (before softmax). |
| **Softmax** | Turns logits into probabilities that sum to 1. |
| **Loss** | A scalar measuring how wrong the predictions are. |
| **Optimizer** | The rule that updates parameters from gradients (SGD, Adam, AdamW). |
| **Learning rate** | How far the optimizer moves each parameter per step. |
| **Embedding** | A lookup table mapping integer token ids to vectors. |
| **Attention** | A way for each position to gather information from other positions. |
| **Head** | One parallel attention computation. Multi-head attention runs several. |
| **KV cache** | Stored keys/values from previous tokens, used to speed up generation. |
| **MoE** | Mixture of Experts: a router sends each token to a subset of expert networks. |
| **LoRA** | Low-Rank Adaptation: freeze the base weights and train a small low-rank update. |
| **Residual connection** | Adding a layer's input back to its output: `x + f(x)`. |
| **Normalization** | Rescaling activations so training is stable (LayerNorm, RMSNorm, BatchNorm). |
| **Activation** | A nonlinear function applied element-wise (ReLU, SiLU, tanh). |
| **Inference** | Using a trained model without updating parameters. |
| **MIMO** | Multi-input, multi-output model: `fit([x1, x2], [y1, y2])`. |
| **Functional API** | Building a graph with `Input` and calling layers on symbolic tensors. |
| **Sequential API** | A linear stack of layers. |
