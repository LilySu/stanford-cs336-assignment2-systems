from __future__ import annotations

from typing import Type

import torch

import math

# Check if triton is available
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Pure PyTorch implementation of FlashAttention-2 forward pass with tiling.
        """
        # Input shapes: (Batch, Seq_Q, Dim) and (Batch, Seq_K, Dim)
        # We can handle arbitrary leading dimensions by flattening them into a single batch dim.
        # The prompt test uses: (Batch, Seq, Dim)
        
        # Save original shape to restore output later
        q_shape = Q.shape
        batch_dims = q_shape[:-2]
        seq_len_q = q_shape[-2]
        d = q_shape[-1]
        seq_len_k = K.shape[-2]
        
        # Flatten batch dimensions
        q = Q.view(-1, seq_len_q, d)
        k = K.view(-1, seq_len_k, d)
        v = V.view(-1, seq_len_k, d)
        
        batch_size = q.shape[0]
        
        # Hyperparameters for tiling
        # Must be at least 16x16 as per prompt. 
        # Clean powers of 2 simplify logic (no bounds checking needed per prompt guarantee).
        Br = 64  # Block size for Q (Rows)
        Bc = 64  # Block size for K, V (Cols)
        
        # Initialize Output and LogSumExp accumulators
        # O will accumulate the numerator first, then be divided by L at the end
        O = torch.zeros_like(q)
        # L stores the running denominator (sum of exps).
        # We initialize max_score (m) to -inf and sum_exp (l) to 0.
        # But to efficiently use the update rule, we can just store LSE directly or separate m and l.
        # Here we will store the running row-wise max `m` and running row-wise sum `l`.
        m = torch.full((batch_size, seq_len_q), float('-inf'), device=Q.device)
        l = torch.zeros((batch_size, seq_len_q), device=Q.device)
        
        scale = 1.0 / math.sqrt(d)
        
        # Outer loop: Iterate over batches
        for b in range(batch_size):
            # Tiling Loop 1: Iterate over blocks of Q (Rows)
            for i in range(0, seq_len_q, Br):
                q_i = q[b, i : i + Br]  # Shape: (Br, d)
                
                # Load current statistics for this block
                m_i = m[b, i : i + Br]  # Shape: (Br,)
                l_i = l[b, i : i + Br]  # Shape: (Br,)
                o_i = O[b, i : i + Br]  # Shape: (Br, d)
                
                # Tiling Loop 2: Iterate over blocks of K, V (Cols)
                for j in range(0, seq_len_k, Bc):
                    k_j = k[b, j : j + Bc]  # Shape: (Bc, d)
                    v_j = v[b, j : j + Bc]  # Shape: (Bc, d)
                    
                    # 1. Compute Score: S_ij = Q_i @ K_j^T * scale
                    # Shape: (Br, Bc)
                    s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale
                    
                    # (Causal masking would go here, but prompt says we can ignore it)
                    
                    # 2. Compute max of current block: m_ij
                    m_ij, _ = torch.max(s_ij, dim=-1) # Shape: (Br,)
                    
                    # 3. Update running max: m_new = max(m_i, m_ij)
                    m_new = torch.maximum(m_i, m_ij)
                    
                    # 4. Compute exponentials with numerical stability
                    # P_ij = exp(S_ij - m_new)
                    p_ij = torch.exp(s_ij - m_new.unsqueeze(-1))
                    
                    # 5. Compute correction factor for old accumulators
                    # alpha = exp(m_i - m_new)
                    alpha = torch.exp(m_i - m_new)
                    
                    # 6. Update sum_exp (l)
                    # l_new = l_i * alpha + rowsum(P_ij)
                    l_new = l_i * alpha + p_ij.sum(dim=-1)
                    
                    # 7. Update Output (O)
                    # O_new = O_i * alpha + P_ij @ V_j
                    # Note: O_i currently holds the unnormalized weighted sum
                    o_new = o_i * alpha.unsqueeze(-1) + torch.matmul(p_ij, v_j)
                    
                    # Write back to accumulators
                    m_i = m_new
                    l_i = l_new
                    o_i = o_new

                # After iterating over all columns, finalize the block row
                # Save stats back to global tensors
                m[b, i : i + Br] = m_i
                l[b, i : i + Br] = l_i
                O[b, i : i + Br] = o_i

        # Final normalization: O = O / l
        O = O / l.unsqueeze(-1)
        
        # Restore original shape
        O = O.view(q_shape)
        
        # Compute final LogSumExp for backward pass: L = m + log(l)
        L = m + torch.log(l)
        L = L.view(batch_dims + (seq_len_q,))
        
        # Save for backward (as requested in prompt)
        ctx.save_for_backward(L, Q, K, V, O)
        
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    return FlashAttentionFunction


# ==========================================
# Part B & C: Triton Implementation
# ==========================================

if triton is not None:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr, 
    ):
        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Q Block Pointer
        Q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # K Block Pointer
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # V Block Pointer
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        # O Block Pointer (accumulator)
        O_block_ptr = tl.make_block_ptr(
            base=O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # Initialize Accumulators
        m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        acc_o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        # Load Q
        q = tl.load(Q_block_ptr)
        
        # Range for K loop
        for start_k in range(0, N_KEYS, K_TILE_SIZE):
            # Load K and V
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
            
            # 1. Compute Scores
            qk = tl.dot(q, tl.trans(k))
            qk *= scale
            
            # 2. Causal Masking
            if IS_CAUSAL:
                # Construct index vectors
                offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                offs_k = start_k + tl.arange(0, K_TILE_SIZE)
                
                # Compare to form mask (Bq x Bk)
                mask = offs_q[:, None] >= offs_k[None, :]
                
                # Apply mask: add constant value of -1e6 (or overwrite with it)
                qk = tl.where(mask, qk, -1.0e6)

            # 3. Standard FlashAttention Update
            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            
            l_i = l_i * alpha + tl.sum(p, 1)
            
            acc_o = acc_o * alpha[:, None]
            p = p.to(v.dtype)
            acc_o = tl.dot(p, v, acc=acc_o)
            
            m_i = m_new
            
            # Update pointers! (Must assign back to variable)
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        # Finalize Output
        acc_o = acc_o / l_i[:, None]
        tl.store(O_block_ptr, acc_o.to(O_ptr.dtype.element_ty))
        
        # Store L
        offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        L_ptr_curr = L_ptr + batch_index * stride_lb + offs_q * stride_lq
        L_val = m_i + tl.log(l_i)
        tl.store(L_ptr_curr, L_val)


class TritonFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Shape check
        q_shape = Q.shape
        batch_size = q_shape[0]
        seq_len_q = q_shape[1]
        seq_len_k = K.shape[1]
        d = q_shape[2]
        
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, seq_len_q), device=Q.device, dtype=torch.float32)
        
        # Reduced block size to 64 to fit shared memory
        BLOCK_Q = 64
        BLOCK_K = 64
        
        grid = (triton.cdiv(seq_len_q, BLOCK_Q), batch_size)
        scale = 1.0 / math.sqrt(d)
        
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_len_q, seq_len_k,
            scale,
            D=d,
            Q_TILE_SIZE=BLOCK_Q,
            K_TILE_SIZE=BLOCK_K,
            IS_CAUSAL=is_causal,
        )
        
        # Save variables and the causal flag
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")


def get_flashattention_autograd_function_triton() -> Type:
    if triton is None:
        raise ImportError("Triton is not available.")
    return TritonFlashAttentionFunction


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    raise NotImplementedError


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
