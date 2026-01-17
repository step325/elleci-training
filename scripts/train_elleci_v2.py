"""
Elleci V2 - Training Script with 3-Phase Curriculum Learning

Architecture: Elleci v2-MoE (~4B params, ~1.5B active)
- DifferentialMamba2Block + MoE-FFN (even layers)
- EGMLASelfAttention + Dense SwiGLU (odd layers)
- 8 experts, top-2 routing with DPSL load balancing

Training: 60K steps (35K EN + 15K IT + 10K Instruct)
- Phase 1: English Foundation (FineWeb-Edu, Cosmopedia, Math, Code)
- Phase 2: Italian Knowledge (CulturaX, Wikipedia IT)
- Phase 3: Instruction Alignment (OpenOrca, Fauno, Alpaca IT)

Usage:
    python scripts/train_elleci_v2.py
    python scripts/train_elleci_v2.py --dry-run
    python scripts/train_elleci_v2.py --resume checkpoints/elleci_v2_step_10000.pth
"""
import os
import sys
import math
import time
import gc
import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import SWALR
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# Add root to path
sys.path.append(os.getcwd())

from src.config import ElleciConfig, TrainingConfigV2
from src.model import Elleci
from data.elleci_dataset_v2 import EllediDatasetV2

# ======== MAXIMUM CUDA PERFORMANCE OPTIMIZATIONS ========
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(False)
# =========================================================

# Optional dependencies
BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    pass


class MockWandB:
    run = None
    def init(self, **kwargs):
        print(f"WandB not found. Logging disabled.")
    def log(self, data, step=None):
        pass
    def finish(self):
        pass


class ManualSWA:
    """
    Manual SWA implementation that doesn't use deepcopy.
    Avoids RuntimeError with non-leaf tensors (e.g., from BitNet).
    """
    def __init__(self, model):
        self.n_averaged = 0
        # Clone state dict to float for numerical stability
        self.avg_state_dict = {
            k: v.clone().detach().float()
            for k, v in model.state_dict().items()
        }

    def update_parameters(self, model):
        """Update running average with current model parameters."""
        self.n_averaged += 1
        with torch.no_grad():
            for k, v in model.state_dict().items():
                # Running average: avg = avg + (new - avg) / n
                self.avg_state_dict[k] += (v.float() - self.avg_state_dict[k]) / self.n_averaged

    def state_dict(self):
        """Return averaged state dict."""
        return {k: v.clone() for k, v in self.avg_state_dict.items()}

    def apply_to_model(self, model):
        """Apply averaged weights to model."""
        model.load_state_dict(self.avg_state_dict)


if wandb is None:
    wandb = MockWandB()


def parse_args():
    parser = argparse.ArgumentParser(description="Elleci v2 Training")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--start-step", type=int, default=None, help="Step to resume from")
    parser.add_argument("--phase", type=int, default=None, help="Start from specific phase (1, 2, or 3)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for MoE)")
    parser.add_argument("--grad-accum", type=int, default=32, help="Gradient accumulation steps")
    return parser.parse_args()


def get_lr_schedule(step, total_steps, warmup_steps, cooldown_ratio=0.2):
    """WSD Scheduler: Warmup-Stable-Decay"""
    cooldown_start = int(total_steps * (1 - cooldown_ratio))

    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    elif step < cooldown_start:
        return 1.0
    else:
        cooldown_steps = total_steps - cooldown_start
        cooldown_progress = (step - cooldown_start) / cooldown_steps
        return 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))


def get_current_seq_len(step, total_steps, phase, training_config):
    """Get sequence length based on curriculum schedule."""
    if phase == 3:
        schedule = training_config.instruction_curriculum
    else:
        schedule = training_config.curriculum_schedule

    # Calculate progress within current phase
    if phase == 1:
        phase_progress = step / training_config.phase1_steps
    elif phase == 2:
        phase2_start = training_config.phase1_steps
        phase_progress = (step - phase2_start) / training_config.phase2_steps
    else:
        phase3_start = training_config.phase1_steps + training_config.phase2_steps
        phase_progress = (step - phase3_start) / training_config.phase3_steps

    phase_progress = max(0.0, min(1.0, phase_progress))

    for threshold, seq_len in schedule:
        if phase_progress < threshold:
            return seq_len
    return schedule[-1][1]


def get_current_phase(step, training_config):
    """Determine current training phase based on step."""
    if step < training_config.phase1_steps:
        return 1
    elif step < training_config.phase1_steps + training_config.phase2_steps:
        return 2
    else:
        return 3


def get_batch_size_for_seq_len(seq_len, base_batch_size=2):
    """Scale batch size inversely with sequence length."""
    if seq_len <= 256:
        return base_batch_size * 4
    elif seq_len <= 512:
        return base_batch_size * 2
    elif seq_len <= 1024:
        return base_batch_size
    else:
        return max(1, base_batch_size // 2)


def get_grad_accum_for_seq_len(seq_len, base_accum=32, target_effective=64):
    """Adjust gradient accumulation to maintain effective batch size."""
    batch_size = get_batch_size_for_seq_len(seq_len)
    # target = batch_size * grad_accum
    return max(1, target_effective // batch_size)


def create_lerac_param_groups(model, base_lr=3e-4, warmup_ratio=5.0, weight_decay=0.1):
    """
    LeRaC: Learning Rate Curriculum with selective weight decay.
    Higher LR for early layers, lower for deep layers.
    """
    param_groups = []
    n_layers = len(model.blocks)

    def should_decay(name):
        if 'bias' in name:
            return False
        if 'norm' in name.lower() or 'ln' in name.lower():
            return False
        if 'emb' in name.lower():
            return False
        return True

    # Embeddings: base LR, no weight decay
    emb_params = list(model.token_emb.parameters())
    if emb_params:
        param_groups.append({
            'params': emb_params,
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'embeddings'
        })

    # Blocks: decreasing LR from early to deep
    for i, block in enumerate(model.blocks):
        layer_ratio = 1.0 + (n_layers - i - 1) * (warmup_ratio - 1.0) / max(1, n_layers - 1)
        layer_lr = base_lr * layer_ratio

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

    # Output head
    head_decay = [p for n, p in model.lm_head.named_parameters()
                  if p.requires_grad and should_decay(n)]
    head_no_decay = [p for n, p in model.lm_head.named_parameters()
                     if p.requires_grad and not should_decay(n)]
    head_no_decay += list(model.norm_f.parameters())

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


def cleanup_old_checkpoints(checkpoint_dir="checkpoints", keep_last=3):
    """Keep only the last N checkpoints."""
    if not os.path.exists(checkpoint_dir):
        return

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "elleci_v2_step_*.pth"))
    if len(checkpoints) <= keep_last:
        return

    checkpoints.sort(key=os.path.getmtime)
    to_delete = checkpoints[:-keep_last]
    for ckpt in to_delete:
        try:
            os.remove(ckpt)
            print(f"Deleted old checkpoint: {os.path.basename(ckpt)}")
        except Exception as e:
            print(f"Failed to delete {ckpt}: {e}")


def print_vram_breakdown(model, batch_size, seq_len, d_model, n_layers=24):
    """Show VRAM usage breakdown."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_gb = param_bytes / 1024**3
    grad_gb = param_gb
    optim_gb = (sum(p.numel() for p in model.parameters()) * 2) / 1024**3
    activation_per_layer = batch_size * seq_len * d_model * 2
    activation_gb = (activation_per_layer * n_layers * 2) / 1024**3
    activation_checkpointed_gb = activation_gb / 4
    total_gb = param_gb + grad_gb + optim_gb + activation_checkpointed_gb

    print(f"""
+------------------------------------------+
|           VRAM BREAKDOWN                 |
+------------------------------------------+
| Model weights:       {param_gb:>6.2f} GB         |
| Gradients:           {grad_gb:>6.2f} GB         |
| Optimizer (8-bit):   {optim_gb:>6.2f} GB         |
| Activations (ckpt):  {activation_checkpointed_gb:>6.2f} GB         |
+------------------------------------------+
| TOTAL ESTIMATED:     {total_gb:>6.2f} GB         |
+------------------------------------------+
    """)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"| PyTorch Allocated:   {allocated:>6.2f} GB         |")
        print(f"| PyTorch Reserved:    {reserved:>6.2f} GB         |")
        print("+------------------------------------------+")


def get_moe_aux_loss(model):
    """Collect auxiliary losses from MoE layers."""
    aux_loss = 0.0
    count = 0
    for block in model.blocks:
        if hasattr(block, 'get_aux_loss'):
            loss = block.get_aux_loss()
            if loss is not None:
                aux_loss += loss
                count += 1
    return aux_loss / max(1, count) if count > 0 else None


@torch.no_grad()
def evaluate_model(model, tokenizer, config, phase, seq_len, num_samples=50, dtype=torch.bfloat16):
    """
    Evaluate model on validation samples.

    Returns:
        dict with eval_loss and eval_perplexity
    """
    model.eval()

    # Create a small eval dataset
    eval_dataset = EllediDatasetV2(
        tokenizer,
        phase=phase,
        max_length=seq_len,
        batch_size=2,
        seed=9999  # Different seed for eval
    )
    eval_loader = DataLoader(eval_dataset, batch_size=None, num_workers=0)
    eval_iter = iter(eval_loader)

    total_loss = 0.0
    total_tokens = 0
    samples_processed = 0

    try:
        for _ in range(num_samples):
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_loader)
                batch = next(eval_iter)

            batch = batch.to(config.device)
            targets = batch.clone()

            with torch.amp.autocast('cuda', dtype=dtype):
                _, loss = model(batch, targets=targets)

            if loss is not None:
                batch_tokens = (batch != tokenizer.pad_token_id).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                samples_processed += 1

    except Exception as e:
        print(f"Eval error: {e}")

    model.train()

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))  # Clamp to avoid overflow
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_samples': samples_processed
        }

    return None


def train():
    args = parse_args()

    # Training configuration
    training_config = TrainingConfigV2()

    if args.dry_run:
        print("DRY RUN MODE")
        training_config.phase1_steps = 10
        training_config.phase2_steps = 5
        training_config.phase3_steps = 5
        training_config.save_interval = 5
        training_config.eval_interval = 5

    TOTAL_STEPS = training_config.total_steps
    WARMUP_STEPS = int(TOTAL_STEPS * training_config.warmup_ratio)
    SWA_START = int(TOTAL_STEPS * training_config.swa_start_ratio)

    print("Elleci V2-MoE Training")
    print("=" * 60)
    print(f"Total Steps: {TOTAL_STEPS}")
    print(f"  Phase 1 (EN): {training_config.phase1_steps} steps")
    print(f"  Phase 2 (IT): {training_config.phase2_steps} steps")
    print(f"  Phase 3 (Instr): {training_config.phase3_steps} steps")
    print(f"Warmup: {WARMUP_STEPS} steps")
    print(f"SWA Start: {SWA_START} steps")

    # Load tokenizer
    tokenizer_path = "tokenizer_elleci_v1/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "tokenizer_chimera_v2_patched/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"

    vocab_len = len(tokenizer.get_vocab())
    rounded_vocab = ((vocab_len + 127) // 128) * 128
    print(f"Vocab: {vocab_len} -> Rounded: {rounded_vocab}")

    # Model configuration
    d_model = 2048
    n_layers = 24

    if args.dry_run:
        d_model = 768
        n_layers = 4
        print(f"DRY RUN: Using tiny model ({d_model}d x {n_layers}L)")

    config = ElleciConfig(
        vocab_size=rounded_vocab,
        max_seq_len=2048,  # Support up to 2048 for instruction phase
        d_model=d_model,
        n_layers=n_layers,
        use_router=False,
        use_v2=True,      # Enable DifferentialMamba + EG-MLA
        use_moe=True,     # Enable MoE
        use_4bit_act=False,  # No BitNet a4.8
        batch_size=args.batch_size,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    # Ensure Mamba-2 is enabled
    config.mamba.use_mamba2 = True

    # MoE configuration
    config.moe.num_experts = 8
    config.moe.top_k = 2
    config.moe.aux_loss_weight = 0.01

    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nArchitecture: Elleci v2-MoE")
    print(f"  use_v2: {config.use_v2} (DifferentialMamba + EG-MLA)")
    print(f"  use_moe: {config.use_moe} (8 experts, top-2)")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")

    # Check VRAM
    if config.device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU VRAM: {vram:.2f} GB")

    # Create model
    model = Elleci(config).to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e9:.2f}B")
    print(f"Trainable params: {trainable_params / 1e9:.2f}B")

    # Resume from checkpoint
    start_step = 0
    resume_optimizer = None
    resume_scheduler = None
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=config.device)

            # Handle both old (state_dict only) and new (full checkpoint) formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format with full checkpoint
                model.load_state_dict(checkpoint['model_state_dict'])
                resume_optimizer = checkpoint.get('optimizer_state_dict')
                resume_scheduler = checkpoint.get('scheduler_state_dict')
                start_step = checkpoint.get('step', 0)
                print(f"Loaded full checkpoint from step {start_step}")
                if 'config' in checkpoint:
                    ckpt_config = checkpoint['config']
                    print(f"  Checkpoint config: d_model={ckpt_config.get('d_model')}, "
                          f"n_layers={ckpt_config.get('n_layers')}, use_v2={ckpt_config.get('use_v2')}")
            else:
                # Old format (just state_dict)
                model.load_state_dict(checkpoint)
                if args.start_step is not None:
                    start_step = args.start_step
                else:
                    import re
                    match = re.search(r'step_(\d+)', args.resume)
                    if match:
                        start_step = int(match.group(1))
                        print(f"Auto-detected start step: {start_step}")

            print(f"Resuming from step {start_step}")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print(f"Checkpoint not found: {args.resume}")
            sys.exit(1)

    # Compile (optional)
    if args.compile and hasattr(torch, "compile") and config.device == "cuda":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    # Optimizer with LeRaC
    lr = config.learning_rate
    param_groups = create_lerac_param_groups(model, base_lr=lr, warmup_ratio=1.5)

    print(f"\nLeRaC enabled ({len(param_groups)} groups)")
    for pg in param_groups[:3]:
        print(f"  {pg['name']}: LR = {pg['lr']:.2e}")

    if BNB_AVAILABLE:
        print("Using 8-bit AdamW (bitsandbytes)")
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.1)
    else:
        print("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_schedule(step, TOTAL_STEPS, WARMUP_STEPS)
    )

    # Load optimizer/scheduler states if resuming from full checkpoint
    if resume_optimizer is not None:
        try:
            optimizer.load_state_dict(resume_optimizer)
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("  Continuing with fresh optimizer...")

    if resume_scheduler is not None:
        try:
            scheduler.load_state_dict(resume_scheduler)
            print("Loaded scheduler state from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
            # Fallback: advance scheduler manually
            if start_step > 0:
                print(f"  Advancing scheduler to step {start_step}...")
                for _ in range(start_step):
                    scheduler.step()
    elif start_step > 0:
        # Old checkpoint format: advance scheduler manually
        print(f"Advancing scheduler to step {start_step}...")
        for _ in range(start_step):
            scheduler.step()

    # Mixed precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Precision: {'bfloat16' if use_bf16 else 'float16'}")

    scaler = torch.amp.GradScaler('cuda', enabled=(config.device == 'cuda' and not use_bf16))

    # SWA (lazy init)
    swa_model = None
    swa_scheduler = None
    swa_active = False

    # WandB
    if args.no_wandb:
        print("WandB disabled")
        os.environ["WANDB_MODE"] = "disabled"
    elif wandb and not args.dry_run:
        try:
            wandb.init(
                entity="stefano-crepaldi98-universit-degli-studi-di-ferrara",
                project="Elleci",
                name="elleci-v2-moe",
                config={
                    "model": "Elleci v2-MoE",
                    "d_model": config.d_model,
                    "n_layers": config.n_layers,
                    "use_v2": config.use_v2,
                    "use_moe": config.use_moe,
                    "moe_experts": config.moe.num_experts,
                    "moe_top_k": config.moe.top_k,
                    "total_steps": TOTAL_STEPS,
                    "lr": lr,
                }
            )
        except Exception as e:
            print(f"WandB init failed: {e}")

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

    # Clear CUDA cache
    if config.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Determine initial phase and settings
    current_phase = args.phase if args.phase else get_current_phase(start_step, training_config)
    current_seq_len = get_current_seq_len(start_step, TOTAL_STEPS, current_phase, training_config)
    current_batch_size = get_batch_size_for_seq_len(current_seq_len, args.batch_size)
    current_grad_accum = get_grad_accum_for_seq_len(current_seq_len, args.grad_accum)

    print(f"\nStarting Phase {current_phase}")
    print(f"  seq_len: {current_seq_len}")
    print(f"  batch_size: {current_batch_size}")
    print(f"  grad_accum: {current_grad_accum}")
    print(f"  effective_batch: {current_batch_size * current_grad_accum}")

    # Create dataset
    dataset = EllediDatasetV2(
        tokenizer,
        phase=current_phase,
        max_length=current_seq_len,
        batch_size=current_batch_size
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)
    data_iter = iter(dataloader)

    # Training loop
    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"\nStarting training from step {start_step}...")
    pbar = tqdm(range(start_step, TOTAL_STEPS), initial=start_step, total=TOTAL_STEPS)

    accum_loss = 0
    accum_aux_loss = 0
    first_step_done = False

    for step in pbar:
        # Check phase transition
        new_phase = get_current_phase(step, training_config)
        if new_phase != current_phase:
            gc.collect()
            torch.cuda.empty_cache()

            current_phase = new_phase
            print(f"\n>>> PHASE {current_phase} started at step {step}")

            # Update dataset
            current_seq_len = get_current_seq_len(step, TOTAL_STEPS, current_phase, training_config)
            current_batch_size = get_batch_size_for_seq_len(current_seq_len, args.batch_size)
            current_grad_accum = get_grad_accum_for_seq_len(current_seq_len, args.grad_accum)

            dataset = EllediDatasetV2(
                tokenizer,
                phase=current_phase,
                max_length=current_seq_len,
                batch_size=current_batch_size
            )
            dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)
            data_iter = iter(dataloader)

        # Check curriculum transition within phase
        new_seq_len = get_current_seq_len(step, TOTAL_STEPS, current_phase, training_config)
        if new_seq_len != current_seq_len:
            gc.collect()
            torch.cuda.empty_cache()

            current_seq_len = new_seq_len
            current_batch_size = get_batch_size_for_seq_len(current_seq_len, args.batch_size)
            current_grad_accum = get_grad_accum_for_seq_len(current_seq_len, args.grad_accum)
            print(f"\nCURRICULUM: seq_len={current_seq_len}, batch={current_batch_size}, accum={current_grad_accum}")

            dataset = EllediDatasetV2(
                tokenizer,
                phase=current_phase,
                max_length=current_seq_len,
                batch_size=current_batch_size
            )
            dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)
            data_iter = iter(dataloader)

        # SWA activation
        if step == SWA_START and not swa_active:
            print(f"\nSWA activated at step {step}")
            swa_model = ManualSWA(model)
            swa_scheduler = SWALR(optimizer, swa_lr=lr * training_config.swa_lr_factor)
            swa_active = True
            torch.cuda.empty_cache()

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(config.device)
        targets = batch.clone()

        # Forward pass
        with torch.amp.autocast('cuda', dtype=dtype):
            logits, loss = model(batch, targets=targets)

            # Add MoE auxiliary loss
            moe_aux = get_moe_aux_loss(model)
            if moe_aux is not None:
                loss = loss + moe_aux
                accum_aux_loss += moe_aux.item() / current_grad_accum

        # Backward pass
        if use_bf16:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        accum_loss += loss.item() / current_grad_accum

        # VRAM breakdown on first step
        if not first_step_done:
            print_vram_breakdown(model, current_batch_size, current_seq_len, d_model, n_layers)
            first_step_done = True

        # Optimizer step
        if (step + 1) % current_grad_accum == 0:
            # Scale gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.div_(current_grad_accum)

            if use_bf16:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            # SWA update
            if swa_active:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            # Logging
            curr_lr = scheduler.get_last_lr()[0]
            if wandb.run:
                log_data = {
                    "train/loss": accum_loss,
                    "train/lr": curr_lr,
                    "train/phase": current_phase,
                    "train/seq_len": current_seq_len,
                }
                if accum_aux_loss > 0:
                    log_data["train/moe_aux_loss"] = accum_aux_loss
                wandb.log(log_data, step=step)

            pbar.set_description(
                f"Loss: {accum_loss:.4f} | LR: {curr_lr:.2e} | Ph: {current_phase} | Seq: {current_seq_len}"
            )
            accum_loss = 0
            accum_aux_loss = 0

            # Periodic cleanup
            if (step + 1) % 50 == 0 and config.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

        # Evaluation
        if (step + 1) % training_config.eval_interval == 0:
            print(f"\n[Eval] Running evaluation at step {step + 1}...")
            eval_results = evaluate_model(
                model, tokenizer, config,
                phase=current_phase,
                seq_len=current_seq_len,
                num_samples=30,
                dtype=dtype
            )

            if eval_results:
                eval_loss = eval_results['eval_loss']
                eval_ppl = eval_results['eval_perplexity']
                print(f"[Eval] Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")

                if wandb.run:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/perplexity": eval_ppl,
                    }, step=step)

                # Clear CUDA cache after eval
                if config.device == "cuda":
                    torch.cuda.empty_cache()

        # Save checkpoint
        if (step + 1) % training_config.save_interval == 0:
            checkpoint_path = f"checkpoints/elleci_v2_step_{step+1}.pth"
            try:
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': step + 1,
                    'phase': current_phase,
                    'seq_len': current_seq_len,
                    'config': {
                        'd_model': config.d_model,
                        'n_layers': config.n_layers,
                        'vocab_size': config.vocab_size,
                        'use_v2': config.use_v2,
                        'use_moe': config.use_moe,
                    },
                }
                # Save SWA model if active
                if swa_active and swa_model is not None:
                    checkpoint_data['swa_state_dict'] = swa_model.state_dict()

                torch.save(checkpoint_data, checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
                cleanup_old_checkpoints(keep_last=training_config.keep_last_checkpoints)
            except RuntimeError as e:
                print(f"\nFailed to save checkpoint: {e}")

    # Final saves
    print("\nTraining complete!")
    torch.save(model.state_dict(), "elleci_v2_final.pth")
    print("Model saved: elleci_v2_final.pth")

    if swa_active and swa_model is not None:
        torch.save(swa_model.state_dict(), "elleci_v2_swa_final.pth")
        print("SWA model saved: elleci_v2_swa_final.pth")

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    train()
