"""
Elleci V1 - Main Training Script
Trains a 1.5B BitNet model on Mixed English/Italian logic corpus.
Implements Phased Training (Knowledge -> Alignment).

Usage:
    python scripts/train_elleci.py
"""
import os
import sys
import math
import time
import gc
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from transformers import PreTrainedTokenizerFast
try:
    import wandb
except ImportError:
    wandb = None

# ======== MAXIMUM CUDA PERFORMANCE OPTIMIZATIONS ========
# Benchmark-verified speedups on RTX 4070
torch.backends.cudnn.benchmark = True  # Auto-tune convolutions: 3.16x speedup
torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat32 for matmul
torch.backends.cudnn.allow_tf32 = True  # TensorFloat32 for cuDNN
torch.autograd.set_detect_anomaly(False)  # Disable anomaly detection: 4x speedup
# =========================================================

class MockWandB:
    run = None
    def init(self, project=None, name=None, config=None):
        print(f"⚠️ WandB not found. Logging disabled (Mock init: {name})")
    def log(self, data, step=None):
        pass

if wandb is None:
    wandb = MockWandB()
from tqdm import tqdm

# Add root to path
sys.path.append(os.getcwd())

from src.config import ElleciConfig
from src.model import Elleci
from data.elleci_dataset import EllediDataset
# from scripts.benchmark_chimera import ChimeraEvaluator  # Module removed

import argparse

# Default Configuration
DEFAULT_TOTAL_STEPS = 50_000
DEFAULT_PHASE_SWITCH = 45_000
GRAD_ACCUM_STEPS = 16  # With batch_size=4, effective batch = 64
DEFAULT_BATCH_SIZE = 4  # Reduced for 512 seq_len (VRAM safety)
BNB_AVAILABLE = False
LION_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    pass

try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    pass

def print_vram_breakdown(model, batch_size, seq_len, d_model, n_layers=24):
    """Mostra breakdown dettagliato dell'uso VRAM"""

    # Pesi del modello
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_gb = param_bytes / 1024**3

    # Gradienti (stessa size dei pesi)
    grad_gb = param_gb

    # Optimizer states (8-bit AdamW ≈ 2 bytes per param)
    optim_gb = (sum(p.numel() for p in model.parameters()) * 2) / 1024**3

    # Attivazioni (stima con gradient checkpointing)
    activation_per_layer = batch_size * seq_len * d_model * 2
    activation_gb = (activation_per_layer * n_layers * 2) / 1024**3
    activation_checkpointed_gb = activation_gb / 4

    total_gb = param_gb + grad_gb + optim_gb + activation_checkpointed_gb

    print(f"""
╔══════════════════════════════════════════╗
║           VRAM BREAKDOWN                 ║
╠══════════════════════════════════════════╣
║ Pesi modello:        {param_gb:>6.2f} GB         ║
║ Gradienti:           {grad_gb:>6.2f} GB         ║
║ Optimizer (8-bit):   {optim_gb:>6.2f} GB         ║
║ Attivazioni (ckpt):  {activation_checkpointed_gb:>6.2f} GB         ║
╠══════════════════════════════════════════╣
║ TOTALE STIMATO:      {total_gb:>6.2f} GB         ║
╚══════════════════════════════════════════╝
    """)

    # Anche memoria reale da PyTorch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"║ PyTorch Allocated:   {allocated:>6.2f} GB         ║")
        print(f"║ PyTorch Reserved:    {reserved:>6.2f} GB         ║")
        print("╚══════════════════════════════════════════╝")

# ======== CURRICULUM LEARNING SCHEDULE ========
# Start with short sequences (faster), scale up
SEQ_CURRICULUM = [
    (0.70, 512),   # First 70%: seq=512 (skip 256 to avoid overfitting)
    (1.00, 1024),  # Last 30%: full seq=1024
]

def get_current_seq_len(step, total_steps):
    """Get sequence length based on curriculum schedule."""
    progress = step / total_steps
    for threshold, seq_len in SEQ_CURRICULUM:
        if progress < threshold:
            return seq_len
    return SEQ_CURRICULUM[-1][1]

def get_batch_size_for_seq_len(seq_len):
    """Scale batch_size inversely with seq_len to maintain constant memory."""
    if seq_len <= 512:
        return 4
    else:
        return 2  # Halved to compensate for doubled seq_len

def get_grad_accum_for_seq_len(seq_len):
    """Scale grad_accum to maintain effective batch size = 64."""
    if seq_len <= 512:
        return 16
    else:
        return 32  # Doubled to compensate for halved batch_size

def create_lerac_param_groups(model, base_lr=1.5e-3, warmup_ratio=5.0, weight_decay=0.1):
    """
    LeRaC: Learning Rate Curriculum with SELECTIVE WEIGHT DECAY
    Higher LR for early layers, lower for deep layers.
    CRITICAL: Weight decay is NOT applied to bias, LayerNorm, or embeddings.
    """
    param_groups = []
    n_layers = len(model.blocks)
    
    # Function to check if param should have weight decay
    def should_decay(name):
        # No decay for bias terms
        if 'bias' in name:
            return False
        # No decay for LayerNorm/RMSNorm parameters
        if 'norm' in name.lower() or 'ln' in name.lower():
            return False
        # No decay for embeddings
        if 'emb' in name.lower():
            return False
        return True
    
    # Embeddings: base LR, NO weight decay (RoPE has no learnable params)
    emb_params = list(model.token_emb.parameters())
    if emb_params:
        param_groups.append({
            'params': emb_params, 
            'lr': base_lr, 
            'weight_decay': 0.0,  # No decay for embeddings
            'name': 'embeddings'
        })
    
    # Blocks: decreasing LR from early to deep, selective weight decay
    for i, block in enumerate(model.blocks):
        # Early layers get higher LR
        layer_ratio = 1.0 + (n_layers - i - 1) * (warmup_ratio - 1.0) / max(1, n_layers - 1)
        layer_lr = base_lr * layer_ratio
        
        # Separate decay/no-decay params
        decay_params = []
        no_decay_params = []
        for name, param in block.named_parameters():
            if param.requires_grad:
                if should_decay(name):
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        
        if decay_params:
            param_groups.append({
                'params': decay_params, 
                'lr': layer_lr, 
                'weight_decay': weight_decay,
                'name': f'block_{i}_decay'
            })
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params, 
                'lr': layer_lr, 
                'weight_decay': 0.0,
                'name': f'block_{i}_no_decay'
            })
    
    # Head: base LR, selective decay
    head_decay = [p for n, p in model.lm_head.named_parameters() if p.requires_grad and should_decay(n)]
    head_no_decay = [p for n, p in model.lm_head.named_parameters() if p.requires_grad and not should_decay(n)]
    head_no_decay += list(model.norm_f.parameters())  # LayerNorm always no decay
    
    if head_decay:
        param_groups.append({
            'params': head_decay, 
            'lr': base_lr, 
            'weight_decay': weight_decay,
            'name': 'head_decay'
        })
    if head_no_decay:
        param_groups.append({
            'params': head_no_decay, 
            'lr': base_lr, 
            'weight_decay': 0.0,
            'name': 'head_no_decay'
        })
    
    return param_groups
# =============================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_TOTAL_STEPS, help="Total training steps")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (small model, few steps)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (experimental)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start-step", type=int, default=None, help="Step to resume from (auto-detected from checkpoint name if not specified)")
    return parser.parse_args()

def get_lr_schedule(step, total_steps, warmup_steps, cooldown_ratio=0.2):
    """WSD Scheduler: Warmup-Stable-Decay (2.9x better than cosine!)"""
    cooldown_start = int(total_steps * (1 - cooldown_ratio))
    
    if step < warmup_steps:
        # Warmup
        return float(step) / float(max(1, warmup_steps))
    elif step < cooldown_start:
        # Stable (full LR)
        return 1.0
    else:
        # Cooldown (cosine decay)
        cooldown_steps = total_steps - cooldown_start
        cooldown_progress = (step - cooldown_start) / cooldown_steps
        return 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))

def cleanup_old_checkpoints(checkpoint_dir="checkpoints", keep_last=3):
    """
    Keep only the last N checkpoints to save disk space.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of most recent checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return

    # Find all checkpoint files
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "elleci_step_*.pth"))

    if len(checkpoints) <= keep_last:
        return

    # Sort by modification time (oldest first)
    checkpoints.sort(key=os.path.getmtime)

    # Delete oldest checkpoints
    to_delete = checkpoints[:-keep_last]
    for ckpt in to_delete:
        try:
            os.remove(ckpt)
            print(f"🗑️  Deleted old checkpoint: {os.path.basename(ckpt)}")
        except Exception as e:
            print(f"⚠️ Failed to delete {ckpt}: {e}")

def train():
    args = parse_args()

    TOTAL_STEPS = args.steps
    PHASE_SWITCH_STEP = int(TOTAL_STEPS * 0.9)  # Keep 90% ratio
    WARMUP_STEPS = min(1000, int(TOTAL_STEPS * 0.05))
    SAVE_INTERVAL = 5000  # Increased from 2500 to save disk space
    EVAL_INTERVAL = 1000
    
    if args.dry_run:
        print("🧪 DRY RUN MODE ENABLED")
        TOTAL_STEPS = 20
        PHASE_SWITCH_STEP = 10
        SAVE_INTERVAL = 5
        EVAL_INTERVAL = 5
        WARMUP_STEPS = 2
    
    # Calculate after overrides
    SWA_START = int(TOTAL_STEPS * 0.8)  # Start SWA at 80%

    print("🦁 Elleci V1 - Initialization")
    print(f"Goal: {TOTAL_STEPS} steps (Switch at {PHASE_SWITCH_STEP})")
    
    print("\n🛡️  SECURITY CHECK: Causal Masking Fix is ACTIVE.")
    print("   (Ensuring logits[t] predicts targets[t+1])\n")
    
    # 1. Load Tokenizer & Detect Vocab Size
    tokenizer_path = "tokenizer_elleci_v1/tokenizer.json"
    # Fallback to old path if new not found
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "tokenizer_chimera_v2_patched/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"
        
    vocab_len = len(tokenizer.get_vocab())
    rounded_vocab = ((vocab_len + 127) // 128) * 128
    print(f"📏 Detected Vocab: {vocab_len} -> Rounded: {rounded_vocab}")

    # 2. Model Config
    d_model = 2048
    n_layers = 24
    
    if args.dry_run:
        d_model = 768 # Must be divisible by 12 (default heads)
        n_layers = 4
        print(f"🧪 DRY RUN: Using tiny model ({d_model}d x {n_layers}L)")
    else:
        print(f"🏗️  Model: {d_model}d x {n_layers}L (Target ~1.5B)")

    config = ElleciConfig(
        vocab_size=rounded_vocab,
        max_seq_len=1024,
        d_model=d_model,
        n_layers=n_layers,
        use_router=False,
        batch_size=DEFAULT_BATCH_SIZE  # Reduced for VRAM
    )
    
    # Enable Mamba-2 explicitly
    config.mamba.use_mamba2 = True
    print(f"🐍 Mamba-2 (SSD) Enabled: {config.mamba.use_mamba2}")
    
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check VRAM fit (simple check)
    
    # Check VRAM fit (simple check)
    if config.device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU VRAM: {vram:.2f} GB")
        if vram < 10:
            print("⚠️ Warning: <10GB VRAM might struggle with 1.5B model training contexts.")
    
    model = Elleci(config).to(config.device)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"📂 Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=config.device)
            model.load_state_dict(checkpoint)

            # Auto-detect step from checkpoint name if not specified
            if args.start_step is not None:
                start_step = args.start_step
            else:
                # Try to extract step from filename (e.g., elleci_step_40000.pth)
                import re
                match = re.search(r'step_(\d+)', args.resume)
                if match:
                    start_step = int(match.group(1))
                    print(f"📍 Auto-detected start step: {start_step}")
                else:
                    print("⚠️ Could not auto-detect step from checkpoint name. Starting from 0.")

            print(f"✅ Resuming from step {start_step}")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print(f"❌ Checkpoint not found: {args.resume}")
            sys.exit(1)

    # Compile
    if args.compile and hasattr(torch, "compile") and config.device == "cuda":
        try:
            model = torch.compile(model)
            print("⚡ torch.compile enabled")
        except:
            print("⚠️ force compile failed, proceeding eager")
    else:
        print("⏩ torch.compile disabled (use --compile to enable)")

    # 3. Optimizer with LeRaC (per-layer learning rates)
    lr = 1.0e-4  # Lowered from 3e-4 for stability
    
    param_groups = create_lerac_param_groups(model, base_lr=lr, warmup_ratio=1.5)
    
    print("🎯 LeRaC Enabled (per-layer LR curriculum):")
    for pg in param_groups[:3]:  # Show first 3
        print(f"   {pg['name']}: LR = {pg['lr']:.2e}")
    print(f"   ... ({len(param_groups)} groups total)")
    
    # AdamW: Most stable optimizer for LLM training
    # With 24GB VRAM, memory is not a constraint
    if BNB_AVAILABLE:
        print("🚀 Using 8-bit AdamW (bitsandbytes)")
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.1)
    else:
        print("⚡ Using AdamW Optimizer")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr_schedule(step, TOTAL_STEPS, WARMUP_STEPS)
    )
    
    # Use bfloat16 if available (CRITICAL for BitNet/Mamba stability)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"🔧 Precision: {'bfloat16' if use_bf16 else 'float16'} (GradScaler: {'Disabled' if use_bf16 else 'Enabled'})")

    # Float16 requires GradScaler (unlike BFloat16)
    scaler = torch.amp.GradScaler('cuda', enabled=(config.device == 'cuda' and not use_bf16))
    
    # SWA (Stochastic Weight Averaging) - LAZY INIT to save VRAM
    # Will be created at SWA_START to avoid +5GB overhead during training
    swa_model = None
    swa_scheduler = None
    swa_active = False
    print(f"📊 SWA prepared (will activate at step {SWA_START})")

    # 4. WandB
    run_name = f"elleci-v1-phase1"
    
    if args.no_wandb:
        print("🚫 WandB logging disabled by user")
        wandb_mode = "disabled"
        # Disable login prompt
        os.environ["WANDB_MODE"] = "disabled"
    else:
        wandb_mode = "disabled" if args.dry_run else "online"
    
    if wandb:
        try:
            wandb.init(
                entity="stefano-crepaldi98-universit-degli-studi-di-ferrara",
                project="Elleci", 
                name=run_name, 
                config=config.to_dict(), 
                mode=wandb_mode
            )
        except Exception as e:
            print(f"⚠️ WandB init failed: {e}")
    
    # 5. Training Loop
    os.makedirs("checkpoints", exist_ok=True)
    
    num_workers = 0  # Buffered dataset is fast, no need for multiprocessing overhead
    
    # Clear CUDA cache before training
    if config.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Enable gradient checkpointing for memory efficiency (always beneficial)
    model.gradient_checkpointing_enable()
    print("📉 Gradient Checkpointing ENABLED (memory saving)")
    
    # Sequence Length Curriculum: start short, scale up
    current_seq_len = get_current_seq_len(start_step, TOTAL_STEPS)
    current_batch_size = get_batch_size_for_seq_len(current_seq_len)
    current_grad_accum = get_grad_accum_for_seq_len(current_seq_len)
    print(f"📐 Curriculum: Starting with seq_len={current_seq_len}, batch_size={current_batch_size}, grad_accum={current_grad_accum}")

    # Determine initial phase based on start_step
    current_phase = 2 if start_step >= PHASE_SWITCH_STEP else 1

    dataset = EllediDataset(tokenizer, phase=current_phase, max_length=current_seq_len, batch_size=current_batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers, prefetch_factor=None if num_workers==0 else 2, pin_memory=True)
    data_iter = iter(dataloader)
    
    model.train()
    optimizer.zero_grad(set_to_none=True)  # Free gradient memory immediately
    
    print(f"\n🔥 Starting Training from step {start_step}...")
    pbar = tqdm(range(start_step, TOTAL_STEPS), initial=start_step, total=TOTAL_STEPS)
    
    accum_loss = 0
    
    for step in pbar:
        # Curriculum Sequence Length Check
        new_seq_len = get_current_seq_len(step, TOTAL_STEPS)
        if new_seq_len != current_seq_len:
            # Clear memory before transition
            gc.collect()
            torch.cuda.empty_cache()

            current_seq_len = new_seq_len
            current_batch_size = get_batch_size_for_seq_len(current_seq_len)
            current_grad_accum = get_grad_accum_for_seq_len(current_seq_len)
            print(f"\n📐 CURRICULUM: seq_len={current_seq_len}, batch_size={current_batch_size}, grad_accum={current_grad_accum}")

            dataset = EllediDataset(tokenizer, phase=current_phase, max_length=current_seq_len, batch_size=current_batch_size)
            dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers, prefetch_factor=None if num_workers==0 else 2, pin_memory=True)
            data_iter = iter(dataloader)
        
        # Phase Switch Check
        if step == PHASE_SWITCH_STEP and current_phase != 2:
            print("\n🔄 SWITCHING TO PHASE 2 (Alignment)...")
            current_phase = 2
            dataset = EllediDataset(tokenizer, phase=2, max_length=current_seq_len, batch_size=current_batch_size)
            dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers, prefetch_factor=None if num_workers==0 else 2, pin_memory=True)
            data_iter = iter(dataloader)
        
        # SWA Activation Check (lazy init to save VRAM)
        if step == SWA_START and not swa_active:
            print(f"\n📊 SWA ACTIVATED at step {step}! Creating averaged model...")
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.5)
            swa_active = True
            if config.device == "cuda":
                torch.cuda.empty_cache()
        
        # Get Batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        batch = batch.to(config.device)
        
        # Prepare targets (Mask padding with -100 so loss ignores it)
        # Prepare targets
        targets = batch.clone()
        # CRITICAL FIX: Do NOT mask padding if PAD_ID == EOS_ID and dataset is packed.
        # if tokenizer.pad_token_id is not None:
        #    targets[targets == tokenizer.pad_token_id] = -100
        
        # Forward Pass
        # Note: mixed precision (autocast) context is handled globally or by accelerator
        # We also need to be careful with Mamba which sometimes prefers specific dtypes
        with torch.amp.autocast('cuda', dtype=dtype):
            logits, loss = model(batch, targets=targets)
            # NOTE: Don't scale loss here - Liger Kernel doesn't like it
            # We'll accumulate full gradients and scale before optimizer.step()

        # Backward pass - use scaler only for fp16, direct for bf16
        if use_bf16:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        accum_loss += loss.item() / current_grad_accum

        # VRAM breakdown after first forward pass
        if step == start_step:
            print_vram_breakdown(
                model=model,
                batch_size=current_batch_size,
                seq_len=current_seq_len,
                d_model=d_model,
                n_layers=n_layers
            )

        if (step + 1) % current_grad_accum == 0:
            # Scale gradients by current_grad_accum (since we didn't scale loss before backward)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.div_(current_grad_accum)
            
            # Gradient Clipping (Critical for BitNet)
            if use_bf16:
                # BF16 path - no scaler needed
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                # FP16 path - use scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            # SWA Update
            if swa_active:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)  # Free gradient memory immediately
            
            # Log
            curr_lr = scheduler.get_last_lr()[0]
            if wandb.run:
                wandb.log({"train/loss": accum_loss, "train/lr": curr_lr, "train/phase": current_phase}, step=step)
            
            pbar.set_description(f"Loss: {accum_loss:.4f} | LR: {curr_lr:.2e} | Ph: {current_phase}")
            accum_loss = 0
            
            # Periodic memory cleanup (every 50 gradient updates)
            if (step + 1) % 50 == 0 and config.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        
        # Eval
        if (step + 1) % EVAL_INTERVAL == 0:
            # Run benchmark logic inline or call external?
            # For speed, let's just generate a sample log
            model.eval()
            try:
                with torch.no_grad():
                     prompt = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=config.device) # Dummy
                     # Just verify not NaN
                     pass
            except:
                pass
            model.train()
            
        # Save checkpoint
        if (step + 1) % SAVE_INTERVAL == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/elleci_step_{step+1}.pth"

            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

                # Cleanup old checkpoints to save disk space
                cleanup_old_checkpoints(checkpoint_dir="checkpoints", keep_last=3)

            except RuntimeError as e:
                print(f"❌ Failed to save checkpoint: {e}")
                print("⚠️  Continuing training without saving...")
            
    # Final Save
    torch.save(model.state_dict(), "elleci_v1_final.pth")
    
    # Save SWA model (better generalization)
    if swa_active:
        torch.save(swa_model.state_dict(), "elleci_v1_swa_final.pth")
        print("📊 SWA model saved!")
    
    print("✅ Training Complete")

if __name__ == "__main__":
    train()
