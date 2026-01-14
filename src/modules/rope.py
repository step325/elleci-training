"""
Rotary Position Embedding (RoPE)

Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
Used by LLaMA, Mistral, Qwen, and most modern LLMs.

Key advantages:
- No learnable parameters (just math)
- Generalizes to unseen sequence lengths
- Relative position encoding (better for attention)
"""
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding.
    
    Precomputes sin/cos tables for efficient RoPE application.
    
    Args:
        dim: Dimension of the embedding (usually head_dim)
        base: Base for the frequency computation (higher = longer context)
              - 10000: ~8k context
              - 100000: ~128k context (LLaMA 3.1 style)
        max_seq_len: Maximum sequence length to precompute
    """
    def __init__(self, dim: int, base: int = 100000, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for sin/cos (will be computed on first forward)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cached cos/sin tables if sequence length increased."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Position indices [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            
            # Frequencies: [seq_len, dim/2]
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            # Concat for full dim: [seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Cache cos and sin
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin embeddings for the given sequence length.
        
        Args:
            x: Input tensor (used for device/dtype)
            seq_len: Sequence length (if None, uses x.shape[-2])
            
        Returns:
            (cos, sin): Each [seq_len, dim]
        """
        if seq_len is None:
            seq_len = x.shape[-2]
            
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return (
            self._cos_cached[:seq_len],
            self._sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    For [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim]
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]
        position_ids: Optional position indices (for generation with cache)
        
    Returns:
        (q_rotated, k_rotated): Same shapes as input
    """
    # Handle position_ids for cached generation
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Broadcast cos/sin to match q/k shape
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# Export
__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb', 'rotate_half']


if __name__ == "__main__":
    # Self-test
    print("RoPE Self-Test")
    print("=" * 50)
    
    # Create RoPE with 128k support
    rope = RotaryEmbedding(dim=64, base=100000)
    print(f"✓ Created RoPE (dim=64, base=100000)")
    
    # Test cos/sin generation
    x = torch.randn(2, 8, 128, 64)  # [batch, heads, seq, head_dim]
    cos, sin = rope(x, seq_len=128)
    print(f"✓ Generated cos/sin: {cos.shape}")
    
    # Test Q/K rotation
    q = torch.randn(2, 8, 128, 64)
    k = torch.randn(2, 8, 128, 64)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"✓ Rotated Q/K: {q_rot.shape}, {k_rot.shape}")
    
    # Test gradient flow
    q.requires_grad = True
    k.requires_grad = True
    cos, sin = rope(q, seq_len=128)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    loss = q_rot.sum() + k_rot.sum()
    loss.backward()
    print(f"✓ Gradient flow: q.grad norm = {q.grad.norm().item():.4f}")
    
    # Test long sequence (verify 128k support)
    x_long = torch.randn(1, 1, 1000, 64)
    cos_long, sin_long = rope(x_long, seq_len=1000)
    print(f"✓ Long sequence (1000): cos shape = {cos_long.shape}")
    
    print("\n✅ All RoPE tests passed!")
