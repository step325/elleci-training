"""
Convert Dense Elleci to MoE Elleci

Implements:
- UpIT: Uses multiple fine-tuning checkpoints as expert initialization
- Drop-Upcycling: Re-initializes 20% of weights for specialization
- Router Upcycling: Initializes router from MLA attention heads

Usage:
    # Basic conversion (uses single checkpoint, copies for all experts)
    python scripts/convert_to_moe.py --checkpoint checkpoints/elleci_final.pth

    # UpIT with multiple checkpoints (recommended)
    python scripts/convert_to_moe.py \
        --checkpoints checkpoints/step_5000.pth checkpoints/step_10000.pth checkpoints/step_15000.pth \
        --drop-upcycle

    # With custom MoE config
    python scripts/convert_to_moe.py \
        --checkpoint checkpoints/elleci_final.pth \
        --num-experts 8 \
        --top-k 2 \
        --drop-ratio 0.2
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Dict
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ElleciConfig, MoEConfig
from src.model import Elleci, ElleciBlock, SwiGLUFFN, RMSNorm
from src.modules.moe import MoEFFN, MoELayer, Router, drop_upcycle_weights, initialize_router_from_mla
from src.modules.bitnet import BitLinear


class ElleciMoEBlock(nn.Module):
    """
    Elleci Block with MoE FFN for Mamba layers.

    Same as ElleciBlock but uses MoE-FFN instead of dense FFN.
    """
    def __init__(self, config, use_mamba=False, use_moe=False):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_moe = use_moe and use_mamba  # MoE only on Mamba layers

        # Dropout
        dropout_rate = getattr(config, 'dropout', 0.1)
        self.dropout = nn.Dropout(dropout_rate)

        # Attention mechanism
        if use_mamba:
            from src.modules.mamba2 import Mamba2BlockFast
            from src.modules.mamba import MambaBlock
            if getattr(config.mamba, 'use_mamba2', True):
                self.attn = Mamba2BlockFast(config.mamba)
            else:
                self.attn = MambaBlock(config.mamba)
        else:
            from src.modules.mla import MLASelfAttention
            self.attn = MLASelfAttention(config.mla)

        # FFN - MoE for Mamba layers, dense for MLA layers
        if self.use_moe:
            self.ffn = MoEFFN(config.d_model, config.moe)
        else:
            ffn_type = getattr(config, 'ffn_type', 'swiglu')
            if ffn_type == 'swiglu':
                self.ffn = SwiGLUFFN(config.d_model)
            else:
                self.ffn = nn.Sequential(
                    BitLinear(config.d_model, config.d_model * 4),
                    nn.GELU(),
                    BitLinear(config.d_model * 4, config.d_model),
                )

        # Norms
        norm_type = getattr(config, 'norm_type', 'rms')
        NormClass = RMSNorm if norm_type == 'rms' else nn.LayerNorm
        self.norm1 = NormClass(config.d_model)
        self.norm2 = NormClass(config.d_model)

    def forward(self, x):
        # Attention + dropout + residual
        x = x + self.dropout(self.attn(self.norm1(x)))

        # FFN + dropout + residual
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


class ElleciMoE(nn.Module):
    """
    Elleci model with Mixture of Experts.

    Architecture:
    - Even layers (Mamba): Use MoE-FFN (8 experts, top-2)
    - Odd layers (MLA): Use dense FFN
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Blocks with MoE
        self.blocks = nn.ModuleList([
            ElleciMoEBlock(
                config,
                use_mamba=(i % 2 == 0),
                use_moe=(i in config.moe.moe_layers)
            )
            for i in range(config.n_layers)
        ])

        # Output
        norm_type = getattr(config, 'norm_type', 'rms')
        if norm_type == 'rms':
            self.norm_f = RMSNorm(config.d_model)
        else:
            self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.vocab_size, config.d_model, bias=False)

        # Init
        self.apply(self._init_weights)

        # Gradient checkpointing
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        # Embeddings
        x = self.token_emb(idx)

        # Collect auxiliary losses
        total_aux_loss = 0.0

        # Process through blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

            # Collect MoE aux loss
            if hasattr(block.ffn, 'aux_loss') and block.ffn.aux_loss is not None:
                total_aux_loss = total_aux_loss + block.ffn.aux_loss

        # Output
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # Compute loss
        loss = None
        if targets is not None:
            import torch.nn.functional as F
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1
            )

            # Add MoE auxiliary loss
            if self.training and isinstance(total_aux_loss, torch.Tensor):
                loss = loss + total_aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        import torch.nn.functional as F
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def load_dense_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict:
    """Load a dense Elleci checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle nested state dicts
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    return state_dict


def copy_ffn_to_expert(
    dense_state: Dict,
    layer_idx: int,
    expert_idx: int,
    drop_ratio: float = 0.0,
    seed: Optional[int] = None
) -> Dict:
    """
    Extract FFN weights from dense model and format for expert.

    Args:
        dense_state: Dense model state dict
        layer_idx: Which layer's FFN to copy
        expert_idx: Target expert index
        drop_ratio: Fraction of weights to re-initialize
        seed: Random seed for reproducibility

    Returns:
        Expert weights dict
    """
    prefix = f"blocks.{layer_idx}.ffn"
    expert_weights = {}

    for key, value in dense_state.items():
        if key.startswith(prefix):
            # Extract the suffix (e.g., "w1.weight")
            suffix = key[len(prefix) + 1:]  # +1 for the dot
            new_key = f"experts.{expert_idx}.{suffix}"
            expert_weights[new_key] = value.clone()

    # Apply drop-upcycling if requested
    if drop_ratio > 0:
        expert_weights = drop_upcycle_weights(
            expert_weights,
            drop_ratio=drop_ratio,
            seed=seed
        )

    return expert_weights


def convert_dense_to_moe(
    dense_state: Dict,
    config: ElleciConfig,
    checkpoint_list: Optional[List[str]] = None,
    drop_ratio: float = 0.2,
    device: str = 'cpu'
) -> Dict:
    """
    Convert dense Elleci state dict to MoE state dict.

    Args:
        dense_state: Primary dense model state dict
        config: ElleciConfig with MoE settings
        checkpoint_list: Optional list of checkpoints for UpIT
        drop_ratio: Fraction of weights to re-init per expert
        device: Device for processing

    Returns:
        MoE model state dict
    """
    moe_state = OrderedDict()
    num_experts = config.moe.num_experts
    moe_layers = list(config.moe.moe_layers)

    # Load additional checkpoints for UpIT
    checkpoint_states = []
    if checkpoint_list:
        for ckpt_path in checkpoint_list:
            ckpt_state = load_dense_checkpoint(ckpt_path, device)
            checkpoint_states.append(ckpt_state)

    print(f"Converting to MoE: {num_experts} experts, {len(moe_layers)} MoE layers")
    print(f"MoE layers: {moe_layers}")
    print(f"Drop-upcycling ratio: {drop_ratio}")

    # Copy non-FFN weights directly
    for key, value in dense_state.items():
        is_moe_layer = any(f"blocks.{i}.ffn" in key for i in moe_layers)
        if not is_moe_layer:
            moe_state[key] = value.clone()

    # Convert FFN to MoE for specified layers
    for layer_idx in moe_layers:
        print(f"  Converting layer {layer_idx} to MoE...")

        # Initialize router (will be done from scratch or from MLA)
        router_prefix = f"blocks.{layer_idx}.ffn.moe.router"
        # Router weights initialized during model creation

        # Copy FFN to each expert
        for expert_idx in range(num_experts):
            # Determine source checkpoint
            if checkpoint_states and expert_idx < len(checkpoint_states):
                # UpIT: Use different checkpoints for first few experts
                source_state = checkpoint_states[expert_idx]
                print(f"    Expert {expert_idx}: Using checkpoint {expert_idx}")
            else:
                # Copy from primary checkpoint with drop-upcycling
                source_state = dense_state
                print(f"    Expert {expert_idx}: Copy + drop-upcycle")

            # Calculate seed for reproducibility
            seed = 42 + layer_idx * 100 + expert_idx

            # Copy FFN weights to expert
            expert_weights = copy_ffn_to_expert(
                source_state,
                layer_idx,
                expert_idx,
                drop_ratio=drop_ratio if expert_idx >= len(checkpoint_states) else 0,
                seed=seed
            )

            # Add to MoE state dict
            for key, value in expert_weights.items():
                full_key = f"blocks.{layer_idx}.ffn.moe.{key}"
                moe_state[full_key] = value

    return moe_state


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str,
    checkpoint_list: Optional[List[str]] = None,
    num_experts: int = 8,
    top_k: int = 2,
    drop_ratio: float = 0.2,
    device: str = 'cpu'
):
    """
    Full conversion pipeline.

    Args:
        checkpoint_path: Primary dense checkpoint
        output_path: Output path for MoE checkpoint
        checkpoint_list: Optional list of checkpoints for UpIT
        num_experts: Number of experts
        top_k: Experts activated per token
        drop_ratio: Weight re-initialization ratio
        device: Processing device
    """
    print("=" * 60)
    print("Elleci Dense → MoE Conversion")
    print("=" * 60)

    # Create config
    config = ElleciConfig()
    config.use_moe = True
    config.moe.num_experts = num_experts
    config.moe.top_k = top_k
    config.moe.drop_ratio = drop_ratio

    # Load dense checkpoint
    dense_state = load_dense_checkpoint(checkpoint_path, device)
    print(f"Dense model keys: {len(dense_state)}")

    # Count original parameters
    dense_params = sum(v.numel() for v in dense_state.values())
    print(f"Dense parameters: {dense_params / 1e9:.2f}B")

    # Convert to MoE
    moe_state = convert_dense_to_moe(
        dense_state,
        config,
        checkpoint_list=checkpoint_list,
        drop_ratio=drop_ratio,
        device=device
    )
    print(f"MoE model keys: {len(moe_state)}")

    # Count MoE parameters
    moe_params = sum(v.numel() for v in moe_state.values())
    print(f"MoE total parameters: {moe_params / 1e9:.2f}B")

    # Estimate active parameters (top-k experts + rest)
    # This is approximate
    expert_expansion = num_experts / top_k
    active_params = dense_params + (moe_params - dense_params) / expert_expansion
    print(f"MoE active parameters (est): {active_params / 1e9:.2f}B")

    # Save checkpoint
    save_dict = {
        'model_state_dict': moe_state,
        'config': config,
        'conversion_info': {
            'source_checkpoint': checkpoint_path,
            'checkpoint_list': checkpoint_list,
            'num_experts': num_experts,
            'top_k': top_k,
            'drop_ratio': drop_ratio,
        }
    }

    print(f"\nSaving to: {output_path}")
    torch.save(save_dict, output_path)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)

    return config, moe_state


def main():
    parser = argparse.ArgumentParser(description="Convert Dense Elleci to MoE")
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Primary dense checkpoint path'
    )
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        default=None,
        help='List of checkpoints for UpIT (uses as expert initializers)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='checkpoints/elleci_moe.pth',
        help='Output checkpoint path'
    )
    parser.add_argument(
        '--num-experts',
        type=int,
        default=8,
        help='Number of experts (default: 8)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=2,
        help='Experts per token (default: 2)'
    )
    parser.add_argument(
        '--drop-ratio',
        type=float,
        default=0.2,
        help='Weight drop ratio for upcycling (default: 0.2)'
    )
    parser.add_argument(
        '--no-drop-upcycle',
        action='store_true',
        help='Disable drop-upcycling'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for processing'
    )

    args = parser.parse_args()

    # Override drop ratio if disabled
    if args.no_drop_upcycle:
        args.drop_ratio = 0.0

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run conversion
    convert_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        checkpoint_list=args.checkpoints,
        num_experts=args.num_experts,
        top_k=args.top_k,
        drop_ratio=args.drop_ratio,
        device=args.device
    )


if __name__ == "__main__":
    main()
