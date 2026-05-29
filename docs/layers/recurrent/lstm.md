# Long Short-Term Memory (LSTM)

## Overview
LSTM is a type of recurrent neural network (RNN) architecture designed to solve the vanishing gradient problem in standard RNNs. It uses a gating mechanism to regulate the flow of information.

## Mathematical Formulation
For each time step $t$:
1.  **Forget Gate**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2.  **Input Gate**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
3.  **Cell Candidate**: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
4.  **Cell State Update**: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
5.  **Output Gate**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
6.  **Hidden State**: $h_t = o_t * \tanh(C_t)$

## Implementation Details
The `LSTM` layer in `neutro` performs the full forward and backward pass (Backpropagation Through Time, BPTT) over the sequence dimension. We concatenate the weights for the four gates into a single matrix to optimize the dot product operations.

## Citations
- Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*. [DOI: 10.1162/neco.1997.9.8.1735](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)
- [Original Paper PDF (Bioinf JKU)](https://www.bioinf.jku.at/publications/older/2604.pdf)
