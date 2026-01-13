"""
NanoPrime v2.0 - Complete Integrated Model

Hybrid architecture combining:
- BitNet 1.58b (ternary weights)
- MLA (KV cache compression)
- Mamba (state space model)
- Adaptive Router (Fast/Slow paths)
- Thinking Loop (recurrent reasoning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from modules.bitnet import BitLinear
from modules.mla import MLASelfAttention
from modules.mamba import MambaBlock
from modules.mamba2 import Mamba2BlockFast
from modules.router import AdaptiveRouter
from modules.thinking_loop import ThinkingLoop

# Liger Kernel for fused operations (memory + speed optimization)
# NOTE: Fused CE disabled due to Triton bug with vocab 32128 + bf16
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    LIGER_AVAILABLE = True
    print("🐯 Liger Kernel available (fused CE disabled for stability)")
except ImportError:
    LIGER_AVAILABLE = False


class NanoPrimeBlock(nn.Module):
    """
    Single transformer block with hybrid Mamba/MLA attention.
    
    Can use either MLA or Mamba based on configuration.
    """
    def __init__(self, config, use_mamba=False):
        super().__init__()
        self.use_mamba = use_mamba
        
        # Attention mechanism
        if use_mamba:
            # Use Mamba-2 if enabled in config, else fallback to v1
            if getattr(config.mamba, 'use_mamba2', True):
                self.attn = Mamba2BlockFast(config.mamba)
            else:
                self.attn = MambaBlock(config.mamba)
        else:
            self.attn = MLASelfAttention(config.mla)
        
        # Feed-forward network (using BitLinear)
        # Note: No LayerNorm here - already applied via norm2 before FFN
        self.ffn = nn.Sequential(
            BitLinear(config.d_model, config.d_model * 4),
            nn.GELU(),
            BitLinear(config.d_model * 4, config.d_model),
        )
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x):
        # Attention + residual
        x = x + self.attn(self.norm1(x))
        
        # FFN + residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class NanoPrime(nn.Module):
    """
    Complete NanoPrime model with adaptive routing.
    
    Architecture:
    - Input embedding
    - Router decides path per input
    - Fast path: 2 shallow blocks
    - Slow path: Thinking loop (4 iterations)
    - Output head
    
    Args:
        config: NanoPrimeConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        
        # Router (Optional)
        if config.use_router:
            self.router = AdaptiveRouter(config.router)
            
            # Fast path: 2 lightweight blocks
            self.fast_blocks = nn.ModuleList([
                NanoPrimeBlock(config, use_mamba=False) for _ in range(2)
            ])
            
            # Slow path: Single block for thinking loop
            self.slow_block = NanoPrimeBlock(config, use_mamba=True)
            self.thinking_loop = ThinkingLoop(config.thinking, self.slow_block)
        else:
            self.router = None
            self.fast_blocks = None
            self.slow_block = None
            self.thinking_loop = None
        
        # Main blocks (shared)
        self.blocks = nn.ModuleList([
            NanoPrimeBlock(config, use_mamba=(i % 2 == 0))  # Alternate Mamba/MLA
            for i in range(config.n_layers)
        ])
        
        # Output
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights (optional)
        # self.lm_head.weight = self.token_emb.weight
        
        # Initialize
        # Initialize
        self.apply(self._init_weights)

        # Gradient Checkpointing flag
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.
        """
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        """
        self.gradient_checkpointing = False
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: Token indices [batch, seq_len]
            targets: Target tokens [batch, seq_len] (for training)
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: (Optional) cross-entropy loss
        """
        batch_size, seq_len = idx.shape
        
        # Embeddings
        tok_emb = self.token_emb(idx)  # [batch, seq, d_model]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)  # [seq, d_model]
        
        x = tok_emb + pos_emb
        
        # Router decision (if enabled)
        load_balance_loss = None
        
        if self.config.use_router and self.router is not None:
            if self.training:
                # ===== TOKEN-LEVEL SOFT ROUTING (CRITICAL FIX) =====
                # Get routing probabilities PER TOKEN (causal!)
                route_probs = self.router(x)  # [batch, seq, 2]
                
                # Extract slow path probability for each token
                p_slow = route_probs[:, :, 1].unsqueeze(-1)  # [batch, seq, 1]
                p_fast = 1.0 - p_slow  # [batch, seq, 1]
                
                # Load balancing loss (aggregate over all tokens)
                target_slow_ratio = 0.3
                actual_slow_ratio = route_probs[:, :, 1].mean()  # Mean over batch AND sequence!
                load_balance_loss = 0.01 * (actual_slow_ratio - target_slow_ratio).pow(2)
                
                # ===== SOFT MIXING: Compute BOTH paths =====
                # Fast path
                x_fast = x.clone()
                for block in self.fast_blocks:
                    x_fast = block(x_fast)
                
                # Slow path (thinking loop processes whole sequence)
                x_slow, _ = self.thinking_loop(x)
                
                # Weighted sum (DIFFERENTIABLE!)
                # Gradient flows to router based on which path helped more
                x = (x_slow * p_slow) + (x_fast * p_fast)  # [batch, seq, d_model]
            else:
                # ===== INFERENCE: Optional hard routing for speed =====
                # For now, use soft mixing for stability
                route_probs = self.router(x)
                p_slow = route_probs[:, :, 1].unsqueeze(-1)
                p_fast = 1.0 - p_slow
                
                x_fast = x.clone()
                for block in self.fast_blocks:
                    x_fast = block(x_fast)
                x_slow, _ = self.thinking_loop(x)
                
                x = (x_slow * p_slow) + (x_fast * p_fast)
        else:
            # Default: use main blocks without routing
            # Default: use main blocks without routing
            for block in self.blocks:
                if self.gradient_checkpointing and self.training:
                    # Use checkpointing to save VRAM (trades compute for memory)
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        
        # Output normalization (logits computed conditionally below)
        x = self.norm_f(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            if LIGER_AVAILABLE and self.training:
                # 🔴 DISABILITIAMO SOLO IL FUSED CE (bug Triton)
                logits = self.lm_head(x)
                main_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
                logits = None  # Not computed in fused mode
            else:
                # Standard path (inference or fallback)
                logits = self.lm_head(x)  # [batch, seq, vocab_size]
                main_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
            
            # Add load balancing loss if router was used
            if load_balance_loss is not None:
                loss = main_loss + load_balance_loss
            else:
                loss = main_loss
        else:
            logits = self.lm_head(x)  # Only compute logits if no targets (inference)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if not None)
            
        Returns:
            Generated tokens [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


import torch.nn.functional as F

# Export
__all__ = ['NanoPrime', 'NanoPrimeBlock']


if __name__ == "__main__":
    # Self-test
    print("NanoPrime Model Self-Test")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, '..')
    from config import NanoPrimeConfig
    
    config = NanoPrimeConfig()
    config.n_layers = 4  # Smaller for testing
    config.max_seq_len = 64
    
    print(f"Config: {config.n_layers} layers, d_model={config.d_model}")
    
    model = NanoPrime(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params/1e6:.1f}M parameters")
    
    # Test forward
    idx = torch.randint(0, config.vocab_size, (2, 32))  # [batch=2, seq=32]
    logits, _ = model(idx, use_router=False)
    print(f"✓ Forward pass: {idx.shape} → {logits.shape}")
    
    # Test with loss
    targets = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx, targets=targets)
    print(f"✓ With targets: loss={loss.item():.4f}")
    
    # Test generation
    start_tokens = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start_tokens, max_new_tokens=10)
    print(f"✓ Generation: {start_tokens.shape} → {generated.shape}")
    
    # Test gradient flow
    loss.backward()
    print(f"✓ Gradient flow: token_emb grad norm = {model.token_emb.weight.grad.norm().item():.4f}")
    
    print("\n✅ All tests passed!")
    print("\n🎉 NanoPrime v2.0 architecture complete!")
