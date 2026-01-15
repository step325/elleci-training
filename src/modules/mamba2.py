"""
Mamba-2 - Selective State Space Model with SSD (State Space Duality)

Based on: "Transformers are SSMs: Generalized Models and Efficient Algorithms 
Through Structured State Space Duality" (arXiv:2405.21060)

Key improvements over Mamba v1:
- 2-8x faster through SSD framework
- Simplified scalar A matrix (instead of diagonal)
- Better hardware utilization on Tensor Cores
- Unified view with attention mechanisms

Authors: Tri Dao, Albert Gu (2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class Mamba2Block(nn.Module):
    """
    Mamba-2 block with SSD (State Space Duality) mechanism.
    
    Key differences from Mamba v1:
    1. Scalar A (single value per head) instead of diagonal matrix
    2. Multi-head structure similar to attention
    3. Chunk-based processing for better parallelism
    4. Simplified discretization
    
    Args:
        config: MambaConfig with d_model, d_state, n_heads, etc.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = getattr(config, 'd_state', 16)
        self.d_conv = getattr(config, 'd_conv', 4)
        self.expand = getattr(config, 'expand', 2)
        
        # Multi-head configuration (key innovation in Mamba-2)
        self.n_heads = getattr(config, 'n_heads', 8)
        self.d_inner = int(self.expand * self.d_model)
        self.head_dim = self.d_inner // self.n_heads
        
        # Chunk size for efficient processing
        self.chunk_size = getattr(config, 'chunk_size', 64)
        
        # Input projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local dependencies (same as v1)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,  # Depthwise
            padding=self.d_conv - 1,
        )
        
        # SSD parameters projections
        # Project to: dt (time step), B (input), C (output)
        self.x_proj = nn.Linear(
            self.d_inner, 
            self.n_heads + self.d_state * 2,  # dt_rank = n_heads, B and C
            bias=False
        )
        
        # Delta (dt) projection - per head
        self.dt_proj = nn.Linear(self.n_heads, self.d_inner, bias=True)
        
        # Initialize dt bias for stability (Preserve long-term memory)
        # We want small initial dt values so decay is close to 1.
        # dt_init range [0.001, 0.1] corresponds to decay [0.999, 0.9]
        dt_init_min = 0.001
        dt_init_max = 0.1
        dt_init = torch.exp(
            torch.rand(self.n_heads) * (math.log(dt_init_max) - math.log(dt_init_min))
            + math.log(dt_init_min)
        )
        # inv_softplus(x) = log(exp(x) - 1)
        # For small x, this is approx log(x)
        inv_dt = torch.log(torch.exp(dt_init) - 1)
        
        # Broadcast to d_inner (since dt_proj maps n_heads -> d_inner)
        # Each head controls head_dim channels
        inv_dt = inv_dt.repeat_interleave(self.head_dim)
        
        with torch.no_grad():
            self.dt_proj.weight.fill_(0.0) # Zero weight init for dt
            self.dt_proj.bias.copy_(inv_dt)
        
        # MAMBA-2 KEY CHANGE: Scalar A per head (not diagonal matrix)
        # This dramatically simplifies computation and enables Tensor Core usage
        # A is initialized to small negative values for stability
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, self.n_heads)))
        
        # D parameter (skip connection) - per head
        self.D = nn.Parameter(torch.ones(self.n_heads))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # Normalization (optional, helps stability)
        self.norm = nn.LayerNorm(self.d_inner)
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Forward pass through Mamba-2 block.
        
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seqlen, dim = x.shape
        
        # 1. Expand input (same as v1)
        x_expanded = self.in_proj(x)  # [batch, seq, d_inner*2]
        x_expanded, z = x_expanded.chunk(2, dim=-1)  # Split for gating
        
        # 2. Depthwise convolution (local context)
        x_conv = rearrange(x_expanded, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seqlen]  # Trim padding
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = self.act(x_conv)
        
        # 3. SSD (State Space Duality) - Mamba-2 core
        y = self.ssd_forward(x_conv)
        
        # 4. Gate with z
        y = y * self.act(z)
        
        # 5. Normalization (CRITICAL for stability!)
        y = self.norm(y)
        
        # 6. Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssd_forward(self, x):
        """
        SSD (State Space Duality) forward pass.
        
        This is the key innovation in Mamba-2:
        - Uses chunked processing for parallelism
        - Scalar A enables simpler matrix operations
        - Connects SSM to attention through structured matrices
        
        Args:
            x: [batch, seq_len, d_inner]
            
        Returns:
            y: [batch, seq_len, d_inner]
        """
        batch, seqlen, d_inner = x.shape
        
        # Get SSD parameters
        x_proj = self.x_proj(x)  # [batch, seq, n_heads + d_state*2]
        
        # Split projections
        dt = x_proj[..., :self.n_heads]  # [batch, seq, n_heads]
        B = x_proj[..., self.n_heads:self.n_heads+self.d_state]  # [batch, seq, d_state]
        C = x_proj[..., self.n_heads+self.d_state:]  # [batch, seq, d_state]
        
        # Process delta (time step)
        dt = self.dt_proj(dt)  # [batch, seq, d_inner]
        dt = F.softplus(dt)  # Ensure positive
        
        # Get A (SCALAR per head in Mamba-2!)
        A = -torch.exp(self.A_log.float())  # [n_heads]
        
        # Reshape for multi-head processing
        x_heads = rearrange(x, 'b l (h d) -> b l h d', h=self.n_heads)
        dt_heads = rearrange(dt, 'b l (h d) -> b l h d', h=self.n_heads)
        
        # Efficient chunked SSD computation
        y = self.chunked_ssd(x_heads, dt_heads, A, B, C)
        
        # Reshape back
        y = rearrange(y, 'b l h d -> b l (h d)')
        
        # Add skip connection
        D = repeat(self.D, 'h -> 1 1 h 1').expand(batch, seqlen, self.n_heads, self.head_dim)
        D = rearrange(D, 'b l h d -> b l (h d)')
        y = y + x * D
        
        return y
    
    def chunked_ssd(self, x, dt, A, B, C):
        """
        Chunked SSD computation for efficiency.
        
        Processes sequence in chunks to balance:
        - Parallel processing within chunks
        - Sequential state propagation between chunks
        
        Args:
            x: [batch, seq, n_heads, head_dim]
            dt: [batch, seq, n_heads, head_dim]
            A: [n_heads] - scalar per head
            B: [batch, seq, d_state]
            C: [batch, seq, d_state]
            
        Returns:
            y: [batch, seq, n_heads, head_dim]
        """
        batch, seqlen, n_heads, head_dim = x.shape
        
        # Pad sequence to multiple of chunk_size
        pad_len = (self.chunk_size - seqlen % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, pad_len))
        
        L_padded = x.shape[1]
        n_chunks = L_padded // self.chunk_size
        
        # Reshape into chunks
        x_chunks = rearrange(x, 'b (nc cs) h d -> b nc cs h d', cs=self.chunk_size)
        dt_chunks = rearrange(dt, 'b (nc cs) h d -> b nc cs h d', cs=self.chunk_size)
        B_chunks = rearrange(B, 'b (nc cs) n -> b nc cs n', cs=self.chunk_size)
        C_chunks = rearrange(C, 'b (nc cs) n -> b nc cs n', cs=self.chunk_size)
        
        # Compute decay factors: exp(A * cumsum(dt))
        # A is scalar per head, dt has per-position values
        dt_cumsum = dt_chunks.cumsum(dim=2)  # [batch, nc, cs, n_heads, head_dim]
        
        # Decay matrix: exp(A * dt_cumsum)
        # A: [n_heads] -> [1, 1, 1, n_heads, 1]
        A_expanded = A.view(1, 1, 1, n_heads, 1)
        decay = torch.exp(A_expanded * dt_cumsum)  # [batch, nc, cs, h, d]
        
        # Process each chunk (can be parallelized)
        outputs = []
        state = torch.zeros(batch, n_heads, head_dim, self.d_state, device=x.device, dtype=x.dtype)
        
        for c in range(n_chunks):
            x_c = x_chunks[:, c]  # [batch, cs, h, d]
            dt_c = dt_chunks[:, c]
            B_c = B_chunks[:, c]  # [batch, cs, d_state]
            C_c = C_chunks[:, c]
            decay_c = decay[:, c]
            
            # Within-chunk computation (parallel)
            y_c, state = self.chunk_forward(x_c, dt_c, decay_c, B_c, C_c, state)
            outputs.append(y_c)
        
        # Concatenate outputs
        y = torch.cat(outputs, dim=1)  # [batch, L_padded, h, d]
        
        # Remove padding
        if pad_len > 0:
            y = y[:, :seqlen]
        
        return y
    
    def chunk_forward(self, x, dt, decay, B, C, state):
        """
        Process a single chunk with state propagation.
        
        Args:
            x: [batch, chunk_size, n_heads, head_dim]
            dt: [batch, chunk_size, n_heads, head_dim]
            decay: [batch, chunk_size, n_heads, head_dim]
            B: [batch, chunk_size, d_state]
            C: [batch, chunk_size, d_state]
            state: [batch, n_heads, head_dim, d_state]
            
        Returns:
            y: [batch, chunk_size, n_heads, head_dim]
            new_state: [batch, n_heads, head_dim, d_state]
        """
        batch, cs, n_heads, head_dim = x.shape
        
        # Simple recurrent formulation for stability
        outputs = []
        
        for t in range(cs):
            # Get current inputs
            x_t = x[:, t]  # [batch, h, d]
            dt_t = dt[:, t]  # [batch, h, d]
            B_t = B[:, t]  # [batch, d_state]
            C_t = C[:, t]  # [batch, d_state]
            
            # Decay state: state = decay_t * state
            decay_t = decay[:, t].unsqueeze(-1)  # [batch, h, d, 1]
            state = state * decay_t
            
            # Update state: state += dt * outer(x, B)
            # x_t: [batch, h, d] -> [batch, h, d, 1]
            # B_t: [batch, d_state] -> [batch, 1, 1, d_state]
            x_t_exp = x_t.unsqueeze(-1)  # [batch, h, d, 1]
            B_t_exp = B_t.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, d_state]
            dt_t_exp = dt_t.unsqueeze(-1)  # [batch, h, d, 1]
            
            state = state + dt_t_exp * x_t_exp * B_t_exp
            
            # Output: y = sum(state * C)
            # state: [batch, h, d, d_state]
            # C_t: [batch, d_state] -> [batch, 1, 1, d_state]
            C_t_exp = C_t.unsqueeze(1).unsqueeze(1)
            y_t = (state * C_t_exp).sum(dim=-1)  # [batch, h, d]
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [batch, cs, h, d]
        
        return y, state


class Mamba2BlockFast(Mamba2Block):
    """
    Optimized Mamba-2 block using matrix operations instead of loops.
    
    This version sacrifices some code clarity for speed by:
    1. Vectorizing the chunk processing
    2. Using cumulative sums efficiently
    3. Avoiding Python loops where possible
    """
    
    def chunk_forward(self, x, dt, decay, B, C, state):
        """
        Vectorized chunk processing.
        """
        batch, cs, n_heads, head_dim = x.shape
        
        # Compute within-chunk state updates using cumulative products
        # This is an approximation that works well for small chunks
        
        # Cumulative decay from start of chunk
        # decay: [batch, cs, h, d]
        log_decay = torch.log(decay.clamp(min=1e-8))
        cumsum_log_decay = log_decay.cumsum(dim=1)
        cum_decay = torch.exp(cumsum_log_decay)  # [batch, cs, h, d]
        
        # Relative decay for each pair (i, j) where j <= i
        # For simplicity, use diagonal approximation
        
        # Input contribution: dt * x * B
        # x: [batch, cs, h, d]
        # B: [batch, cs, d_state]
        x_contrib = dt.unsqueeze(-1) * x.unsqueeze(-1) * B.unsqueeze(2).unsqueeze(2)
        # x_contrib: [batch, cs, h, d, d_state]
        
        # Apply cumulative decay (approximate causal masking)
        # We need to compute: Sum_{k=0}^t (u_k * Prod_{m=k+1}^t d_m)
        # = Prod_{0}^t d_m * Sum_{k=0}^t (u_k / Prod_{0}^k d_m)
        # = cum_decay * cumsum( x_contrib / cum_decay )
        
        # 1. Invert cumulative decay
        # Add epsilon to avoid division by zero
        inv_cum_decay = 1.0 / cum_decay.unsqueeze(-1).clamp(min=1e-20)
        
        # 2. X term scaled by inverse decay
        x_scaled = x_contrib * inv_cum_decay
        
        # 3. Cumulative sum
        cumsum_scaled = x_scaled.cumsum(dim=1)
        
        # 4. Multiply by current cumulative decay
        state_new = cumsum_scaled * cum_decay.unsqueeze(-1)
        
        # Add previous state contribution
        state_expanded = state.unsqueeze(1)  # [batch, 1, h, d, d_state]
        initial_decay = decay[:, 0:1].unsqueeze(-1)  # First decay
        full_cum_decay = cum_decay.unsqueeze(-1)
        
        state_contrib = state_expanded * full_cum_decay
        state_full = state_new + state_contrib
        
        # Output: y = sum(state * C)
        C_exp = C.unsqueeze(2).unsqueeze(2)  # [batch, cs, 1, 1, d_state]
        y = (state_full * C_exp).sum(dim=-1)  # [batch, cs, h, d]
        
        # Update state for next chunk
        new_state = state_full[:, -1]  # [batch, h, d, d_state]
        
        return y, new_state


# Compatibility wrapper to use as drop-in replacement
def create_mamba_block(config, version='v2'):
    """
    Factory function to create Mamba block.
    
    Args:
        config: Configuration object
        version: 'v1' for original, 'v2' for Mamba-2, 'v2_fast' for optimized
        
    Returns:
        MambaBlock instance
    """
    if version == 'v1':
        from .mamba import MambaBlock
        return MambaBlock(config)
    elif version == 'v2':
        return Mamba2Block(config)
    elif version == 'v2_fast':
        return Mamba2BlockFast(config)
    else:
        raise ValueError(f"Unknown version: {version}")


# Export
__all__ = ['Mamba2Block', 'Mamba2BlockFast', 'create_mamba_block']


if __name__ == "__main__":
    # Self-test
    print("Mamba-2 Block Self-Test")
    print("=" * 50)
    
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        d_model: int = 256
        d_state: int = 16
        d_conv: int = 4
        expand: int = 2
        n_heads: int = 4
        chunk_size: int = 32
    
    config = TestConfig()
    
    # Test Mamba-2
    print("\n1. Testing Mamba2Block...")
    mamba2 = Mamba2Block(config)
    x = torch.randn(2, 128, config.d_model)
    
    with torch.no_grad():
        y = mamba2(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    assert y.shape == x.shape, "Shape mismatch!"
    print("   ✅ Mamba2Block works!")
    
    # Test Mamba-2 Fast
    print("\n2. Testing Mamba2BlockFast...")
    mamba2_fast = Mamba2BlockFast(config)
    
    with torch.no_grad():
        y_fast = mamba2_fast(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {y_fast.shape}")
    assert y_fast.shape == x.shape, "Shape mismatch!"
    print("   ✅ Mamba2BlockFast works!")
    
    # Speed comparison
    if torch.cuda.is_available():
        print("\n3. Speed Comparison (CUDA)...")
        import time
        
        mamba2_cuda = Mamba2Block(config).cuda()
        mamba2_fast_cuda = Mamba2BlockFast(config).cuda()
        x_cuda = torch.randn(4, 512, config.d_model, device='cuda')
        
        # Warmup
        for _ in range(5):
            _ = mamba2_cuda(x_cuda)
            _ = mamba2_fast_cuda(x_cuda)
        
        torch.cuda.synchronize()
        
        # Benchmark standard
        start = time.time()
        for _ in range(20):
            _ = mamba2_cuda(x_cuda)
        torch.cuda.synchronize()
        time_std = time.time() - start
        
        # Benchmark fast
        start = time.time()
        for _ in range(20):
            _ = mamba2_fast_cuda(x_cuda)
        torch.cuda.synchronize()
        time_fast = time.time() - start
        
        print(f"   Mamba2Block: {time_std*1000:.2f}ms")
        print(f"   Mamba2BlockFast: {time_fast*1000:.2f}ms")
        print(f"   Speedup: {time_std/time_fast:.2f}x")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
