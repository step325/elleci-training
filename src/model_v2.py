"""
NanoPrime v2.1 - Enhanced Architecture for A/B Testing

Includes:
- RMSNorm option
- SwiGLU FFN option
- SquaredReLU FFN option
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

# RMSNorm Implementation
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# SwiGLU Feed-Forward Network
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, expansion_factor=8/3):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        # Ensure hidden_dim is multiple of 256 for efficiency
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
        
        self.w1 = BitLinear(d_model, hidden_dim)  # Gate projection
        self.w2 = BitLinear(d_model, hidden_dim)  # Up projection
        self.w3 = BitLinear(hidden_dim, d_model)  # Down projection

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# Squared ReLU Feed-Forward Network
class SquaredReLUFFN(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        self.w1 = BitLinear(d_model, hidden_dim)
        self.w2 = BitLinear(hidden_dim, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)) ** 2)

class NanoPrimeBlockV2(nn.Module):
    """
    Enhanced transformer block with configurable FFN and Norm.
    """
    def __init__(self, config, use_mamba=False):
        super().__init__()
        self.use_mamba = use_mamba
        
        dropout_rate = getattr(config, 'dropout', 0.1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Attention mechanism
        if use_mamba:
            if getattr(config.mamba, 'use_mamba2', True):
                self.attn = Mamba2BlockFast(config.mamba)
            else:
                self.attn = MambaBlock(config.mamba)
        else:
            self.attn = MLASelfAttention(config.mla)
        
        # Select FFN type
        ffn_type = getattr(config, 'ffn_type', 'gelu')
        if ffn_type == 'swiglu':
            self.ffn = SwiGLUFFN(config.d_model)
        elif ffn_type == 'sqrelu':
            self.ffn = SquaredReLUFFN(config.d_model)
        else: # Default GELU
            self.ffn = nn.Sequential(
                BitLinear(config.d_model, config.d_model * 4),
                nn.GELU(),
                BitLinear(config.d_model * 4, config.d_model),
            )
        
        # Select Norm type
        norm_type = getattr(config, 'norm_type', 'layer')
        if norm_type == 'rms':
            self.norm1 = RMSNorm(config.d_model)
            self.norm2 = RMSNorm(config.d_model)
        else:
            self.norm1 = nn.LayerNorm(config.d_model)
            self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class NanoPrimeV2(nn.Module):
    """
    Complete NanoPrime V2 model with configurable architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding (Token only, RoPE is in MLA)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(getattr(config.mla, 'dropout', 0.1))
        
        # Backbone blocks
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            # Alternating Mamba/MLA
            use_mamba = (i % 2 == 0) # Even layers = Mamba, Odd = MLA (or vice versa based on preference)
                                     # Looking at original code: Layer Impari (Mamba) -> Index 0 is Mamba
            self.layers.append(NanoPrimeBlockV2(config, use_mamba=use_mamba))
            
        # Final Norm
        norm_type = getattr(config, 'norm_type', 'layer')
        if norm_type == 'rms':
            self.norm_f = RMSNorm(config.d_model)
        else:
            self.norm_f = nn.LayerNorm(config.d_model)
            
        # LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight # Weight tying
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, BitLinear):
            # BitLinear has its own init, usually fine, but good to be safe
            pass

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        x = self.token_emb(idx)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            return logits, None # Loss calculated externally usually, or here if needed
        else:
            logits = self.lm_head(x)
            return logits
