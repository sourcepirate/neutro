import numpy as np
from ..base import Layer
from ...utils.rope_utils import precompute_freqs_cis, apply_rotary_emb

class FlashAttention(Layer):
    """
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
    Implements the tiling and online softmax algorithm from Dao et al. (2022).
    
    Args:
        num_heads: Number of attention heads.
        key_dim: Total dimension of all heads.
        block_size_r: Block size for the outer loop (rows of Q).
        block_size_c: Block size for the inner loop (columns of K, V).
        dropout: Dropout probability.
        use_rope: Whether to use Rotary Positional Embeddings.
    """
    def __init__(self, num_heads, key_dim, block_size_r=64, block_size_c=64, dropout=0.0, use_rope=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
        self.block_size_r = block_size_r
        self.block_size_c = block_size_c
        self.dropout_rate = dropout
        self.use_rope = use_rope
        self.scale = 1.0 / np.sqrt(self.head_dim)

    def build(self, input_shape):
        # input_shape: (batch, seq_len, embed_dim)
        self.embed_dim = input_shape[-1]
        self.params['Wq'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
        self.params['Wk'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
        self.params['Wv'] = np.random.randn(self.embed_dim, self.key_dim) * 0.02
        self.params['Wo'] = np.random.randn(self.key_dim, self.embed_dim) * 0.02
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def forward(self, x, mask=None, training=False, kv_cache=None, layer_id=None):
        self.x = x
        self.mask = mask
        batch_size, seq_len, _ = x.shape
        H = self.num_heads
        d = self.head_dim
        K_dim = self.key_dim

        # Project and split heads
        Q = (x @ self.params['Wq']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
        K = (x @ self.params['Wk']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
        V = (x @ self.params['Wv']).reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)

        if self.use_rope:
            # When using cache, we need the total sequence length for RoPE
            total_seq_len = seq_len
            if kv_cache and layer_id in kv_cache.k_cache:
                total_seq_len += kv_cache.k_cache[layer_id].shape[2]
            
            self.freqs_cis = precompute_freqs_cis(self.head_dim, total_seq_len)
            # Apply RoPE only to the new tokens
            # During generation, seq_len is 1, so we take the last freqs_cis
            if seq_len == 1 and total_seq_len > 1:
                f_cis = self.freqs_cis[total_seq_len-1:total_seq_len]
            else:
                f_cis = self.freqs_cis[:seq_len]
                
            Q = apply_rotary_emb(Q, f_cis)
            K = apply_rotary_emb(K, f_cis)

        # Update and get from cache
        if kv_cache is not None and layer_id is not None:
            K, V = kv_cache.update(K, V, layer_id)
            # Update seq_len for the rest of the calculation
            seq_len_kv = K.shape[2]
        else:
            seq_len_kv = seq_len

        self.Q, self.K, self.V = Q, K, V
        
        # Initialize output and statistics
        # Note: O is (batch, H, seq_len_q, d)
        O = np.zeros_like(Q)
        L = np.zeros((batch_size, H, seq_len, 1))
        M = np.full((batch_size, H, seq_len, 1), -np.inf)

        # Brute force tiling for NumPy implementation
        Br = self.block_size_r
        Bc = self.block_size_c
        Tr = (seq_len + Br - 1) // Br
        Tc = (seq_len_kv + Bc - 1) // Bc

        for j in range(Tc):
            j_start, j_end = j * Bc, min((j + 1) * Bc, seq_len_kv)
            Kj = K[:, :, j_start:j_end, :] # (batch, H, Bc, d)
            Vj = V[:, :, j_start:j_end, :] # (batch, H, Bc, d)

            for i in range(Tr):
                i_start, i_end = i * Br, min((i + 1) * Br, seq_len)
                Qi = Q[:, :, i_start:i_end, :] # (batch, H, Br, d)
                Oi = O[:, :, i_start:i_end, :] # (batch, H, Br, d)
                Mi = M[:, :, i_start:i_end, :] # (batch, H, Br, 1)
                Li = L[:, :, i_start:i_end, :] # (batch, H, Br, 1)

                # Compute scores for this tile
                S_ij = self.scale * (Qi @ Kj.transpose(0, 1, 3, 2)) # (batch, H, Br, Bc)
                
                if mask is not None:
                    # Masking: mask is expected to be (seq_len_q, seq_len_kv)
                    m_tile = mask[i_start:i_end, j_start:j_end]
                    S_ij -= 1e9 * m_tile
                
                # Online softmax stats
                m_ij = np.max(S_ij, axis=-1, keepdims=True)
                P_ij = np.exp(S_ij - m_ij)
                l_ij = np.sum(P_ij, axis=-1, keepdims=True)

                M_new = np.maximum(Mi, m_ij)
                
                # Update L and O
                alpha = np.exp(Mi - M_new)
                beta = np.exp(m_ij - M_new)
                
                # Update O and L in place or using indices
                O[:, :, i_start:i_end, :] = alpha * Oi + beta * (P_ij @ Vj)
                M[:, :, i_start:i_end, :] = M_new
                L[:, :, i_start:i_end, :] = alpha * Li + beta * l_ij

        # Final normalization
        O = O / L
        self.O_pre_proj = O
        self.L = L
        self.M = M
        
        # Merge heads and final projection
        O_merged = O.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
        return O_merged @ self.params['Wo']

    def backward(self, grad_output):
        batch_size, seq_len, embed_dim = self.x.shape
        H = self.num_heads
        d = self.head_dim
        K_dim = self.key_dim
        
        # dWo
        O_merged = self.O_pre_proj.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, K_dim)
        self.grads['Wo'] = O_merged.reshape(-1, K_dim).T @ grad_output.reshape(-1, embed_dim)
        
        # dO_pre_proj
        do_merged = grad_output @ self.params['Wo'].T
        do = do_merged.reshape(batch_size, seq_len, H, d).transpose(0, 2, 1, 3)
        
        # D = rowsum(dO * O)
        D = np.sum(do * self.O_pre_proj, axis=-1, keepdims=True)
        
        dQ = np.zeros_like(self.Q)
        dK = np.zeros_like(self.K)
        dV = np.zeros_like(self.V)
        
        Br = self.block_size_r
        Bc = self.block_size_c
        Tr = (seq_len + Br - 1) // Br
        Tc = (seq_len + Bc - 1) // Bc
        
        for j in range(Tc):
            j_start, j_end = j * Bc, min((j + 1) * Bc, seq_len)
            Kj = self.K[:, :, j_start:j_end, :]
            Vj = self.V[:, :, j_start:j_end, :]
            
            dkj = np.zeros_like(Kj)
            dvj = np.zeros_like(Vj)
            
            for i in range(Tr):
                i_start, i_end = i * Br, min((i + 1) * Br, seq_len)
                Qi = self.Q[:, :, i_start:i_end, :]
                doi = do[:, :, i_start:i_end, :]
                Mi = self.M[:, :, i_start:i_end, :]
                Li = self.L[:, :, i_start:i_end, :]
                Di = D[:, :, i_start:i_end, :]
                
                # Recompute A_ij = exp(S_ij - M_i) / L_i
                S_ij = self.scale * (Qi @ Kj.transpose(0, 1, 3, 2))
                if self.mask is not None:
                    m_tile = self.mask[i_start:i_end, j_start:j_end]
                    S_ij -= 1e9 * m_tile
                
                A_ij = np.exp(S_ij - Mi) / Li
                
                dvj += A_ij.transpose(0, 1, 3, 2) @ doi
                dS_ij = A_ij * (doi @ Vj.transpose(0, 1, 3, 2) - Di)
                
                dQ[:, :, i_start:i_end, :] += self.scale * (dS_ij @ Kj)
                dkj += self.scale * (dS_ij.transpose(0, 1, 3, 2) @ Qi)
            
            dK[:, :, j_start:j_end, :] = dkj
            dV[:, :, j_start:j_end, :] = dvj
            
        if self.use_rope:
            dQ = apply_rotary_emb(dQ, np.conj(self.freqs_cis))
            dK = apply_rotary_emb(dK, np.conj(self.freqs_cis))
        
        # Map back to weights
        dq_flat = dQ.transpose(0, 2, 1, 3).reshape(-1, K_dim)
        dk_flat = dK.transpose(0, 2, 1, 3).reshape(-1, K_dim)
        dv_flat = dV.transpose(0, 2, 1, 3).reshape(-1, K_dim)
        x_flat = self.x.reshape(-1, embed_dim)
        
        self.grads['Wq'] = x_flat.T @ dq_flat
        self.grads['Wk'] = x_flat.T @ dk_flat
        self.grads['Wv'] = x_flat.T @ dv_flat
        
        return (dq_flat @ self.params['Wq'].T + dk_flat @ self.params['Wk'].T + dv_flat @ self.params['Wv'].T).reshape(batch_size, seq_len, embed_dim)
