import numpy as np
from ..base import Layer
from ..core.dense import Dense

class MoELayer(Layer):
    """
    Mixture of Experts (MoE) Layer.
    A group of experts where a router decides which expert is best for each token.
    It's basically a group project where only 1 or 2 experts actually do the work.
    """
    def __init__(self, num_experts, top_k, expert_units, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_units = expert_units
        self.experts = []

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        # The Router: A simple linear layer to score each expert
        self.params['router_weight'] = np.random.normal(0, 0.02, (self.input_dim, self.num_experts))
        
        # Initialize experts as sub-layers
        for i in range(self.num_experts):
            e1 = Dense(self.expert_units, activation='relu')
            e1.build(input_shape)
            
            # expert_shape for second layer
            expert_shape = list(input_shape)
            expert_shape[-1] = self.expert_units
            
            e2 = Dense(self.input_dim)
            e2.build(tuple(expert_shape))
            
            self.experts.append([e1, e2])
        
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, x, training=False):
        # x: (batch, seq_len, dim) or (batch, dim)
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
        # indices: (num_tokens, top_k)
        top_k_indices = np.argsort(router_probs, axis=-1)[:, -self.top_k:]
        self.top_k_indices = top_k_indices
        
        # 3. Dispatch to experts and combine results
        final_output = np.zeros_like(self.x_flat)
        self.expert_outputs = {} # Store for backward
        
        # Group tokens by expert for efficiency
        for expert_idx in range(self.num_experts):
            # Find tokens that have this expert in their top-k
            token_indices, _ = np.where(top_k_indices == expert_idx)
            if len(token_indices) == 0:
                continue
            
            # Get those tokens
            tokens = self.x_flat[token_indices]
            
            # Pass through expert MLP
            out = tokens
            for layer in self.experts[expert_idx]:
                out = layer(out, training=training)
            
            self.expert_outputs[expert_idx] = (token_indices, out)
            
            # Weight by router probability
            weights = router_probs[token_indices, expert_idx].reshape(-1, 1)
            final_output[token_indices] += weights * out
                
        return final_output.reshape(self.x_shape)

    def backward(self, grad_output):
        grad_flat = grad_output.reshape(-1, self.input_dim)
        num_tokens = self.x_flat.shape[0]
        
        dx_flat = np.zeros_like(self.x_flat)
        drouter_logits = np.zeros_like(self.router_probs)
        
        # 1. Backprop through experts and router probabilities
        for expert_idx, (token_indices, out) in self.expert_outputs.items():
            # grad for this expert's output
            weights = self.router_probs[token_indices, expert_idx].reshape(-1, 1)
            expert_grad = weights * grad_flat[token_indices]
            
            # grad for router probability: dL/dP_i = dL/dy * E_i(x)
            # sum across the dim to get a scalar per token
            drouter_probs_expert = np.sum(grad_flat[token_indices] * out, axis=-1)
            drouter_logits[token_indices, expert_idx] = drouter_probs_expert
            
            # Backward through expert layers
            curr_grad = expert_grad
            for layer in reversed(self.experts[expert_idx]):
                curr_grad = layer.backward(curr_grad)
            
            dx_flat[token_indices] += curr_grad

        # 2. Backprop through Softmax for router
        # Simplified softmax backward: probs * (dlogits - sum(probs * dlogits))
        drouter_logits = self.router_probs * (drouter_logits - np.sum(self.router_probs * drouter_logits, axis=-1, keepdims=True))
        
        # 3. Router weight gradient
        self.grads['router_weight'] = self.x_flat.T @ drouter_logits
        
        # 4. Add router's contribution to dx
        dx_flat += drouter_logits @ self.params['router_weight'].T
        
        return dx_flat.reshape(self.x_shape)
