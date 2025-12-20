import torch
import math
from typing import Type

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Implements the FlashAttention-2 forward pass using pure PyTorch with tiling.
        
        Args:
            ctx: Autograd context.
            Q: Query tensor of shape (..., seq_len_q, d).
            K: Key tensor of shape (..., seq_len_k, d).
            V: Value tensor of shape (..., seq_len_k, d).
            is_causal: If True, apply causal masking. (Ignored for this implementation).

        Returns:
            O: Output tensor of shape (..., seq_len_q, d).
        """
        # Save original shape to restore later
        # We treat all dimensions except the last two (seq_len, d) as a flattened batch dimension.
        # This handles both (Batch, Seq, Head, Dim) and (Batch, Seq, Dim) cases.
        original_shape = Q.shape
        batch_dims = original_shape[:-2]
        seq_len_q = original_shape[-2]
        d = original_shape[-1]
        seq_len_k = K.shape[-2]

        # Flatten batch dimensions for uniform processing
        # q_flat: (Batch_Size, seq_len_q, d)
        q_flat = Q.view(-1, seq_len_q, d)
        k_flat = K.view(-1, seq_len_k, d)
        v_flat = V.view(-1, seq_len_k, d)
        
        batch_size = q_flat.shape[0]

        # Block sizes (Tiling) - defined as per instructions (power of 2, >= 16)
        Br = 32  # Block size for rows (Queries)
        Bc = 32  # Block size for columns (Keys/Values)

        # Output buffers
        # O: Accumulated output
        # L: LogSumExp for backward pass
        O_flat = torch.zeros_like(q_flat)
        L_flat = torch.zeros((batch_size, seq_len_q), device=Q.device, dtype=torch.float32)

        # Scaling factor 1/sqrt(d)
        softmax_scale = d ** -0.5

        # ----------------------------------------------------------------------
        # FlashAttention-2 Forward Pass Loop
        # ----------------------------------------------------------------------
        
        # Iterate over batches
        for b in range(batch_size):
            q_b = q_flat[b]  # (seq_len_q, d)
            k_b = k_flat[b]  # (seq_len_k, d)
            v_b = v_flat[b]  # (seq_len_k, d)

            # Iterate over row blocks (Queries) of size Br
            for i in range(0, seq_len_q, Br):
                # Define current Query block limits
                i_end = min(i + Br, seq_len_q)
                q_i = q_b[i:i_end]  # shape: (current_br, d)
                current_br = q_i.shape[0]

                # Initialize running statistics for this row block
                # m_i: max score seen so far (initialized to -inf)
                # l_i: sum of exponentials seen so far (initialized to 0)
                # o_i: weighted sum of values (initialized to 0)
                m_i = torch.full((current_br,), float('-inf'), device=Q.device)
                l_i = torch.zeros((current_br,), device=Q.device)
                o_i = torch.zeros((current_br, d), device=Q.device)

                # Iterate over column blocks (Keys/Values) of size Bc
                for j in range(0, seq_len_k, Bc):
                    # Define current Key/Value block limits
                    j_end = min(j + Bc, seq_len_k)
                    k_j = k_b[j:j_end]  # shape: (current_bc, d)
                    v_j = v_b[j:j_end]  # shape: (current_bc, d)
                    
                    # 1. Compute unnormalized attention scores for this block
                    # S_ij = Q_i @ K_j^T * scale
                    s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * softmax_scale # (current_br, current_bc)

                    # (Note: is_causal masking would be applied here if required)

                    # 2. Update Statistics (Online Softmax)
                    # Compute max of current block
                    m_ij, _ = torch.max(s_ij, dim=-1) # (current_br,)

                    # Update global max for the row
                    m_new = torch.maximum(m_i, m_ij)  # (current_br,)

                    # Compute correction factor alpha = exp(m_old - m_new)
                    # This scales the previously accumulated O and l to the new max
                    alpha = torch.exp(m_i - m_new)    # (current_br,)

                    # Compute exponentials for current block: P_ij = exp(S_ij - m_new)
                    p_ij = torch.exp(s_ij - m_new.unsqueeze(1)) # (current_br, current_bc)

                    # 3. Update Accumulators
                    # Update l (sum of exps): l_new = l_old * alpha + rowsum(P_ij)
                    l_i = l_i * alpha + torch.sum(p_ij, dim=-1)

                    # Update O (weighted values): O_new = O_old * alpha + P_ij @ V_j
                    # We broadcast alpha to (current_br, 1) for the element-wise multiply
                    o_i = o_i * alpha.unsqueeze(1) + torch.matmul(p_ij, v_j)

                    # Update m_i for next iteration
                    m_i = m_new

                # 4. Finalize Block Output
                # The accumulated o_i is unnormalized. Normalize by l_i.
                # Avoid division by zero (though l_i shouldn't be 0 unless masked out completely)
                # O_i = o_i / l_i
                O_flat[b, i:i_end] = o_i / l_i.unsqueeze(1)
                
                # Compute LogSumExp L = m_i + log(l_i)
                # This is saved for the backward pass
                L_flat[b, i:i_end] = m_i + torch.log(l_i)

        # Restore original shape
        O = O_flat.view(original_shape)
        # L has one fewer dimension than O/Q (no head_dim/embedding_dim)
        L = L_flat.view(original_shape[:-1])

        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, O, L)

        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented for this task.")


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns the torch.autograd.Function subclass that implements FlashAttention2.
    """
    return FlashAttentionFunction