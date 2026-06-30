import numpy as np
from neutro.autograd import Tensor, GradientTape, ops


def test_mlp_approximation():
    np.random.seed(42)
    N, D, H, OUT = 32, 4, 16, 2
    x_np = np.random.randn(N, D)
    y_np = np.sin(x_np[:, :OUT])

    W1 = np.random.randn(D, H) * 0.1
    b1 = np.zeros(H)
    W2 = np.random.randn(H, OUT) * 0.01
    b2 = np.zeros(OUT)

    learning_rate = 0.01
    losses = []

    for step in range(100):
        W1_t = Tensor(W1)
        b1_t = Tensor(b1)
        W2_t = Tensor(W2)
        b2_t = Tensor(b2)
        x_t = Tensor(x_np)
        y_t = Tensor(y_np)

        with GradientTape() as tape:
            tape.watch(W1_t)
            tape.watch(b1_t)
            tape.watch(W2_t)
            tape.watch(b2_t)
            h = x_t @ W1_t + b1_t
            h = ops.relu(h)
            pred = h @ W2_t + b2_t
            loss = ((pred - y_t) ** 2).mean()
        losses.append(float(loss.data))

        tape.gradient(loss, [W1_t, b1_t, W2_t, b2_t])

        W1 -= learning_rate * W1_t.grad
        b1 -= learning_rate * b1_t.grad
        W2 -= learning_rate * W2_t.grad
        b2 -= learning_rate * b2_t.grad

    final_loss = losses[-1]
    initial_loss = losses[0]
    assert final_loss < initial_loss, \
        f"MLP did not converge: initial={initial_loss:.6f}, final={final_loss:.6f}"
    assert final_loss < 0.5, f"MLP final loss too high: {final_loss:.6f}"
