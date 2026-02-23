import os
from abc import ABC
from typing import Optional

import torch
import torch.nn.functional as F
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache, flash_attn_varlen_func


_FALLBACK_WARNED = False


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _looks_like_ptx_toolchain_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "provided PTX was compiled with an unsupported toolchain" in msg
        or "Unsupported PTX version" in msg
        or "ptxas fatal" in msg
        or "unsupported toolchain" in msg
    )

class AttentionUtils:
    @staticmethod
    def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute attention using vLLM FlashAttention (FA2) varlen interface for B=1.

        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            v: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)

        Returns:
            Attention output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        if _is_truthy_env("SF_FORCE_TORCH_ATTN"):
            return AttentionUtils.torch_attention(q, k, v, causal=True)

        assert q.shape[0] == 1, "Only B=1 supported in current pipeline"
        T = q.shape[2]

        # Pack to varlen format (T, H, D) and build cumulative lengths for B=1.
        q_var = q.transpose(1, 2).squeeze(0)
        k_var = k.transpose(1, 2).squeeze(0)
        v_var = v.transpose(1, 2).squeeze(0)
        cu = torch.tensor([0, T], device=q.device, dtype=torch.int32)

        try:
            out_var = torch.empty_like(q_var)
            out_var = flash_attn_varlen_func(
                q=q_var,
                k=k_var,
                v=v_var,
                max_seqlen_q=T,
                cu_seqlens_q=cu,
                max_seqlen_k=T,
                cu_seqlens_k=cu,
                causal=True,
                fa_version=2,
                out=out_var,
            )
            return out_var.unsqueeze(0).transpose(1, 2)
        except Exception as e:
            if not _looks_like_ptx_toolchain_error(e):
                raise
            global _FALLBACK_WARNED
            if not _FALLBACK_WARNED:
                _FALLBACK_WARNED = True
                print(
                    f"[sparse-frontier] vLLM FlashAttention kernel failed ({e}); falling back to PyTorch SDPA. "
                    "Set SF_FORCE_TORCH_ATTN=1 to force this behavior."
                )
            return AttentionUtils.torch_attention(q, k, v, causal=True)

    @staticmethod
    def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        """Compute attention using PyTorch SDPA.

        Supports GQA by repeating KV heads to match query heads.
        """
        if q.shape[0] != 1:
            raise ValueError(f"Only B=1 supported in current pipeline (got B={q.shape[0]})")

        if k.shape[1] != q.shape[1]:
            if q.shape[1] % k.shape[1] != 0:
                raise ValueError(
                    f"Invalid GQA: num_q_heads={q.shape[1]} not divisible by num_kv_heads={k.shape[1]}"
                )
            group = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        try:
            return F.scaled_dot_product_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=bool(causal),
            )
        except Exception as e:
            if not _looks_like_ptx_toolchain_error(e):
                raise
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                return F.scaled_dot_product_attention(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=bool(causal),
                )

    @staticmethod
    def attention_with_kvcache(
        query: torch.Tensor,  # [1, num_q_heads, head_dim]
        k_cache: torch.Tensor,  # [num_kv_heads, num_blocks, block_size, head_dim]
        v_cache: torch.Tensor,  # [num_kv_heads, num_blocks, block_size, head_dim]
        cache_seqlens: torch.Tensor,  # [num_q_heads]
        out: torch.Tensor,  # [1, num_q_heads, head_dim]
        block_table: Optional[torch.Tensor] = None,  # [num_q_heads, num_blocks_sel]
    ) -> torch.Tensor:
        if not _is_truthy_env("SF_FORCE_TORCH_ATTN"):
            try:
                num_kv_heads, num_blocks, block_size, head_size = k_cache.shape
                flash_attn_with_kvcache(
                    q=query.squeeze(0).unsqueeze(1).unsqueeze(1),
                    k_cache=k_cache.view(num_kv_heads * num_blocks, block_size, 1, head_size),
                    v_cache=v_cache.view(num_kv_heads * num_blocks, block_size, 1, head_size),
                    block_table=block_table,
                    cache_seqlens=cache_seqlens,
                    causal=True,
                    out=out.squeeze(0).unsqueeze(1).unsqueeze(1),
                )
                return out
            except Exception as e:
                if not _looks_like_ptx_toolchain_error(e):
                    raise
                global _FALLBACK_WARNED
                if not _FALLBACK_WARNED:
                    _FALLBACK_WARNED = True
                    print(
                        f"[sparse-frontier] vLLM FlashAttention KV-cache kernel failed ({e}); "
                        "falling back to PyTorch SDPA. Set SF_FORCE_TORCH_ATTN=1 to force this behavior."
                    )

        # Torch SDPA fallback for decode: q_len == 1, so causal masking is not required.
        out.copy_(AttentionUtils._torch_attention_with_kvcache(query, k_cache, v_cache, cache_seqlens, block_table))
        return out

    @staticmethod
    def _torch_attention_with_kvcache(
        query: torch.Tensor,  # [1, num_q_heads, head_dim]
        k_cache: torch.Tensor,  # [num_kv_heads, num_blocks, block_size, head_dim]
        v_cache: torch.Tensor,  # [num_kv_heads, num_blocks, block_size, head_dim]
        cache_seqlens: torch.Tensor,  # [num_q_heads]
        block_table: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if query.shape[0] != 1 or query.ndim != 3:
            raise ValueError(f"Expected query shape [1, num_q_heads, head_dim], got {tuple(query.shape)}")

        num_q_heads = int(query.shape[1])
        num_kv_heads, num_blocks, block_size, head_size = k_cache.shape

        seqlens = cache_seqlens.to(dtype=torch.long)
        if seqlens.numel() != num_q_heads:
            raise ValueError(
                f"cache_seqlens must have one entry per q head (expected {num_q_heads}, got {seqlens.numel()})"
            )

        s_max = int(seqlens.max().item()) if seqlens.numel() > 0 else 0
        if s_max <= 0:
            return torch.zeros_like(query)

        if block_table is None:
            # Dense: take the prefix of each KV head's cache.
            k_flat = k_cache.reshape(num_kv_heads, -1, head_size)
            v_flat = v_cache.reshape(num_kv_heads, -1, head_size)
            if num_q_heads % num_kv_heads != 0:
                raise ValueError(
                    f"Invalid GQA: num_q_heads={num_q_heads} not divisible by num_kv_heads={num_kv_heads}"
                )
            group = num_q_heads // num_kv_heads
            k_flat = k_flat.repeat_interleave(group, dim=0)[:, :s_max]
            v_flat = v_flat.repeat_interleave(group, dim=0)[:, :s_max]
        else:
            if block_table.shape[0] != num_q_heads:
                raise ValueError(
                    f"block_table must have shape [num_q_heads, *], got {tuple(block_table.shape)}"
                )
            k_blocks = k_cache.view(num_kv_heads * num_blocks, block_size, head_size)
            v_blocks = v_cache.view(num_kv_heads * num_blocks, block_size, head_size)
            indices = block_table.to(dtype=torch.long)
            k_flat = k_blocks[indices].reshape(num_q_heads, -1, head_size)
            v_flat = v_blocks[indices].reshape(num_q_heads, -1, head_size)
            if k_flat.shape[1] < s_max:
                raise RuntimeError(
                    f"Selected KV cache too short for requested seqlens: have {k_flat.shape[1]} tokens, need {s_max}"
                )
            k_flat = k_flat[:, :s_max]
            v_flat = v_flat[:, :s_max]

        q = query.unsqueeze(2)  # [1, H, 1, D]
        k = k_flat.unsqueeze(0)  # [1, H, S, D]
        v = v_flat.unsqueeze(0)  # [1, H, S, D]

        positions = torch.arange(s_max, device=query.device)
        mask = positions.view(1, 1, 1, -1) < seqlens.view(1, -1, 1, 1)

        try:
            out = F.scaled_dot_product_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        except Exception as e:
            if not _looks_like_ptx_toolchain_error(e):
                raise
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                out = F.scaled_dot_product_attention(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
        return out.squeeze(2)
    
    @staticmethod
    def reshape_kv_cache(
        kv_cache: torch.Tensor,
        target_block_size: int,
        max_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve keys and values from cache for attention computation.
        
        Args:
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            target_block_size: Target block size for reshaping
            
        Returns:
            k_cache: [num_kv_heads, num_blocks, target_block_size, head_size]
            v_cache: [num_kv_heads, num_blocks, target_block_size, head_size]
        """
        num_blocks, block_size, num_kv_heads, head_size = kv_cache[0].shape
        final_num_blocks = min(max_blocks, (num_blocks * block_size) // target_block_size)
        left_original_num_blocks = (final_num_blocks * target_block_size) // block_size

        k_cache = kv_cache[0, :left_original_num_blocks, :, :].view(num_kv_heads, final_num_blocks, target_block_size, head_size)
        v_cache = kv_cache[1, :left_original_num_blocks, :, :].view(num_kv_heads, final_num_blocks, target_block_size, head_size)

        return k_cache, v_cache


class AbstractAttention(ABC):
    """Base class for attention implementations (both prefilling and KV compression)"""
    def __init__(self):
        self.block_table = None

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Compute attention with pattern-specific masking.
        
        Args:
            queries: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            keys: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            values: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            layer_idx: Index of the current transformer layer
        Returns:
            Attention output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        return AttentionUtils.flash_attention(queries, keys, values)

    def decode(
        self,
        query: torch.Tensor,  # [1, num_heads, head_dim]
        keys: torch.Tensor,   # [1, num_kv_heads, head_dim]
        values: torch.Tensor, # [1, num_kv_heads, head_dim]
        k_cache: torch.Tensor, # [num_kv_heads, num_blocks, block_size, head_dim]
        v_cache: torch.Tensor, # [num_kv_heads, num_blocks, block_size, head_dim]
        tokens_per_head: torch.Tensor,  # [num_heads]
        output: torch.Tensor, # [1, num_heads, head_dim]
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute attention during decoding phase using flash_attn_with_kvcache.
        
        Args:
            query: Query tensor for a single token [1, num_heads, head_dim]
            keys: Key tensor for the current token [1, num_kv_heads, head_dim]
            values: Value tensor for the current token [1, num_kv_heads, head_dim]
            k_cache: Key cache tensor [num_kv_heads, num_blocks, block_size, head_dim]
            v_cache: Value cache tensor [num_kv_heads, num_blocks, block_size, head_dim]
            tokens_per_head: Tensor of sequence lengths per head [num_heads]
            output: Output tensor to store results [1, num_heads, head_dim]
            layer_idx: Index of the current transformer layer
        """
        _, num_q_heads, _ = query.shape
        num_kv_heads, num_blocks, _block_size, _head_size = k_cache.shape

        if self.block_table is None:
            block_indices = (
                torch.arange(num_blocks * num_kv_heads, device=query.device, dtype=torch.int32)
                .reshape(num_kv_heads, num_blocks)
            )
            block_indices = block_indices.repeat(1, num_q_heads // num_kv_heads)
            self.block_table = block_indices.reshape(num_q_heads, num_blocks)
        
        AttentionUtils.attention_with_kvcache(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=tokens_per_head,
            out=output,
            block_table=self.block_table,
        )

    def kv_compress(
        self, 
        queries: torch.Tensor,  # [num_tokens, num_heads, head_dim]
        keys: torch.Tensor,     # [num_tokens, num_kv_heads, head_size] 
        values: torch.Tensor,   # [num_tokens, num_kv_heads, head_size]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache after prefilling (default: no compression)
        
        Returns:
            tuple: (compressed_keys, compressed_values, seq_lens) where:
                - compressed_keys: [num_kv_heads, max_seq_len, head_size]
                - compressed_values: [num_kv_heads, max_seq_len, head_size]
                - seq_lens: [num_kv_heads] tensor with actual sequence length per head
        """
        # Default implementation: no compression, all tokens kept
        seq_lens = torch.full((keys.size(1),), keys.size(0), device=keys.device, dtype=torch.long)
        # Transpose keys and values to match the expected output shape
        keys_t = keys.transpose(0, 1)  # [num_kv_heads, num_tokens, head_size]
        values_t = values.transpose(0, 1)  # [num_kv_heads, num_tokens, head_size]
        return keys_t, values_t, seq_lens

    def preallocate_memory(self, keys: torch.Tensor) -> None:
        """Pre-allocate any memory needed for this attention method.
        
        Called during vLLM's profiling run to ensure memory is accounted for.
        Default implementation does nothing - override in subclasses that need pre-allocation.
        
        Args:
            keys: Sample key tensor to infer device, dtype, num_heads, head_dim
        """
        pass
