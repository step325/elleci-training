"""
NanoPrime v2.0 - Configuration System

Centralized configuration using dataclasses for easy experimentation.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class BitNetConfig:
    """Configuration for BitNet 1.58b layers"""
    eps: float = 1e-5
    use_bias: bool = False
    
    def __post_init__(self):
        assert self.eps > 0, "eps must be positive"


@dataclass
class MLAConfig:
    """Configuration for Multi-Head Latent Attention"""
    d_model: int = 768
    n_heads: int = 12
    kv_lora_rank: int = 128  # Compression rank for KV cache
    rope_base: int = 100000  # RoPE base (100000 = 128k context support)
    dropout: float = 0.0
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads


@dataclass
class MambaConfig:
    """Configuration for Mamba State Space Model (v1 and v2)"""
    d_model: int = 768
    d_state: int = 16        # Full state size for 24GB VRAM
    d_conv: int = 4          # Local convolution width
    expand: int = 2          # Expansion factor
    dt_rank: str = "auto"    # Rank of Δ (auto = ceil(d_model/16))
    # Mamba-2 specific
    n_heads: int = 8         # Number of heads for Mamba-2 SSD
    chunk_size: int = 64     # Chunk size for efficient processing
    use_mamba2: bool = True  # Use Mamba-2 instead of Mamba v1
    
    def __post_init__(self):
        if self.dt_rank == "auto":
            self.dt_rank = max(1, self.d_model // 16)


@dataclass
class RouterConfig:
    """Configuration for Adaptive Router"""
    d_model: int = 768
    n_experts: int = 2       # Fast path vs Slow path
    router_hidden: int = 256  # Hidden size of gating network
    temperature: float = 1.0  # Softmax temperature
    
    
@dataclass
class ThinkingLoopConfig:
    """Configuration for recurrent thinking loop"""
    max_iterations: int = 4   # Maximum thinking iterations
    convergence_threshold: float = 0.01  # Stop if change < threshold
    

@dataclass
class NanoPrimeConfig:
    """Main model configuration"""
    # Model architecture
    d_model: int = 2048  # 1.5B model
    n_layers: int = 24
    vocab_size: int = 32128   # Elleci tokenizer vocab size
    max_seq_len: int = 1024
    
    # Sub-module configs
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    thinking: ThinkingLoopConfig = field(default_factory=ThinkingLoopConfig)
    
    # Architecture flags
    use_router: bool = False  # Default to False (Production v2 backbone)
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "bf16"  # bf16, fp16, or none
    seed: int = 42
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    
    def __post_init__(self):
        # Ensure sub-configs have correct d_model
        self.mla.d_model = self.d_model
        self.mamba.d_model = self.d_model
        self.router.d_model = self.d_model
        
        # Auto-set n_heads based on d_model (head_dim should be 64-128)
        # d_model=768 -> 12 heads (head_dim=64)
        # d_model=2048 -> 16 heads (head_dim=128)
        if self.d_model >= 2048:
            self.mla.n_heads = 16  # head_dim = 128
        elif self.d_model >= 1024:
            self.mla.n_heads = 16  # head_dim = 64
        else:
            self.mla.n_heads = 12  # head_dim = 64 for 768d
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
        }


if __name__ == "__main__":
    # Test config creation
    config = NanoPrimeConfig()
    print("NanoPrime Configuration:")
    print(f"  Model size: {config.d_model}d × {config.n_layers} layers")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Device: {config.device}")
    print(f"  MLA heads: {config.mla.n_heads}")
    print(f"  Mamba state: {config.mamba.d_state}")
    print("✓ Config validated")
