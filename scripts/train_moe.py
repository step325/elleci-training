"""
Elleci MoE - Fine-tuning Script

Fine-tunes the converted MoE model for 3-5K steps.
Focuses on:
1. Router learning (load balancing)
2. Expert specialization (auxiliary losses)
3. Knowledge retention (low LR for shared params)

Usage:
    python scripts/train_moe.py --checkpoint checkpoints/elleci_moe.pth
    python scripts/train_moe.py --checkpoint checkpoints/elleci_moe.pth --steps 5000 --lr 5e-5
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
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(False)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ElleciConfig
from scripts.convert_to_moe import ElleciMoE

# Try to import data module
try:
    from data.elleci_dataset import EllediDataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("Warning: EllediDataset not available")


class MockWandB:
    """Mock WandB for when it's not installed."""
    run = None
    def init(self, **kwargs):
        print(f"WandB not available. Logging disabled.")
    def log(self, data, step=None):
        pass
    def finish(self):
        pass


if wandb is None:
    wandb = MockWandB()


def create_optimizer(model, base_lr, weight_decay=0.01):
    """
    Create optimizer with different learning rates for different components.

    - Router: Higher LR (needs to learn routing fast)
    - Experts: Medium LR (fine-tune from dense)
    - Shared params: Lower LR (preserve knowledge)
    """
    router_params = []
    expert_params = []
    shared_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'router' in name:
            router_params.append(param)
        elif 'expert' in name:
            expert_params.append(param)
        else:
            shared_params.append(param)

    param_groups = [
        {'params': router_params, 'lr': base_lr * 10, 'weight_decay': 0.0},  # High LR, no WD
        {'params': expert_params, 'lr': base_lr * 2, 'weight_decay': weight_decay},  # Medium LR
        {'params': shared_params, 'lr': base_lr, 'weight_decay': weight_decay},  # Low LR
    ]

    # Use 8-bit Adam if available
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups, betas=(0.9, 0.95))
        print("Using 8-bit AdamW")
    except ImportError:
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
        print("Using standard AdamW")

    return optimizer


def create_scheduler(optimizer, total_steps, warmup_steps):
    """
    Create cosine scheduler with warmup.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_moe_checkpoint(checkpoint_path, device='cuda'):
    """Load MoE checkpoint and create model."""
    print(f"Loading MoE checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = ElleciConfig()
        config.use_moe = True

    # Create model
    model = ElleciMoE(config)

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys: {len(missing)}")
        for key in missing[:5]:
            print(f"  - {key}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for key in unexpected[:5]:
            print(f"  - {key}")

    model = model.to(device)

    return model, config


def get_aux_loss(model):
    """Extract auxiliary loss from MoE layers."""
    total_aux = 0.0
    count = 0

    for block in model.blocks:
        if hasattr(block.ffn, 'aux_loss') and block.ffn.aux_loss is not None:
            total_aux = total_aux + block.ffn.aux_loss
            count += 1

    if count > 0:
        return total_aux / count
    return torch.tensor(0.0)


def train_step(model, batch, optimizer, scheduler, scaler, grad_accum_steps, step):
    """Single training step with gradient accumulation."""
    device = next(model.parameters()).device

    input_ids = batch['input_ids'].to(device)
    targets = input_ids.clone()

    # Forward pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits, loss = model(input_ids, targets=targets)

        # Get auxiliary loss
        aux_loss = get_aux_loss(model)

        # Combined loss (main loss already includes aux if enabled in model)
        total_loss = loss / grad_accum_steps

    # Backward pass
    scaler.scale(total_loss).backward()

    # Gradient accumulation
    if (step + 1) % grad_accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    return loss.item(), aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss


def evaluate(model, eval_loader, max_batches=50):
    """Evaluate model on validation set."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_aux = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            targets = input_ids.clone()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model(input_ids, targets=targets)
                aux_loss = get_aux_loss(model)

            total_loss += loss.item()
            total_aux += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            count += 1

    model.train()

    return total_loss / count, total_aux / count


def save_checkpoint(model, optimizer, scheduler, step, path, config=None):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
    }
    if config:
        checkpoint['config'] = config

    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def train(
    checkpoint_path,
    output_dir='checkpoints',
    total_steps=5000,
    batch_size=4,
    grad_accum_steps=16,
    learning_rate=5e-5,
    warmup_steps=200,
    eval_interval=500,
    save_interval=1000,
    log_interval=10,
    max_seq_len=512,
    device='cuda',
    use_wandb=True
):
    """Main training loop."""
    print("=" * 60)
    print("Elleci MoE Fine-tuning")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, config = load_moe_checkpoint(checkpoint_path, device)
    model.gradient_checkpointing_enable()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params / 1e9:.2f}B")

    # Count MoE-specific params
    router_params = sum(p.numel() for n, p in model.named_parameters() if 'router' in n)
    expert_params = sum(p.numel() for n, p in model.named_parameters() if 'expert' in n)
    print(f"Router parameters: {router_params / 1e6:.2f}M")
    print(f"Expert parameters: {expert_params / 1e9:.2f}B")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, learning_rate)
    scheduler = create_scheduler(optimizer, total_steps, warmup_steps)

    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Load tokenizer
    tokenizer_path = 'tokenizer_chimera_v2_patched'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Create dataset
    if DATASET_AVAILABLE:
        train_dataset = EllediDataset(
            tokenizer=tokenizer,
            max_length=max_seq_len,
            phase='phase2',  # Instruction-heavy for MoE fine-tuning
            streaming=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
    else:
        print("Dataset not available. Creating dummy data for testing.")
        # Dummy dataset for testing
        class DummyDataset:
            def __iter__(self):
                while True:
                    yield {'input_ids': torch.randint(0, 32128, (max_seq_len,))}
        train_loader = DataLoader(DummyDataset(), batch_size=batch_size)

    # Initialize WandB
    if use_wandb and wandb is not None:
        wandb.init(
            project='elleci-moe',
            name=f'moe-finetune-{time.strftime("%Y%m%d-%H%M%S")}',
            config={
                'total_steps': total_steps,
                'batch_size': batch_size,
                'grad_accum_steps': grad_accum_steps,
                'effective_batch': batch_size * grad_accum_steps,
                'learning_rate': learning_rate,
                'warmup_steps': warmup_steps,
                'max_seq_len': max_seq_len,
                'num_experts': config.moe.num_experts,
                'top_k': config.moe.top_k,
            }
        )

    # Training loop
    print("\nStarting training...")
    print(f"Total steps: {total_steps}")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")
    print("-" * 60)

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_aux = 0.0

    start_time = time.time()

    for step in tqdm(range(total_steps), desc="Training"):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Training step
        loss, aux_loss = train_step(
            model, batch, optimizer, scheduler,
            scaler, grad_accum_steps, step
        )

        running_loss += loss
        running_aux += aux_loss

        # Logging
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_aux = running_aux / log_interval
            current_lr = scheduler.get_last_lr()[0]

            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed

            print(f"Step {step+1}/{total_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Aux: {avg_aux:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Speed: {steps_per_sec:.2f} steps/s")

            if use_wandb:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/aux_loss': avg_aux,
                    'train/lr': current_lr,
                    'train/steps_per_sec': steps_per_sec,
                }, step=step)

            running_loss = 0.0
            running_aux = 0.0

        # Save checkpoint
        if (step + 1) % save_interval == 0:
            save_path = os.path.join(output_dir, f'elleci_moe_step_{step+1}.pth')
            save_checkpoint(model, optimizer, scheduler, step, save_path, config)

    # Final save
    final_path = os.path.join(output_dir, 'elleci_moe_final.pth')
    save_checkpoint(model, optimizer, scheduler, total_steps, final_path, config)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: {final_path}")
    print("=" * 60)

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Elleci MoE")
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='MoE checkpoint path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='checkpoints',
        help='Output directory'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=5000,
        help='Total training steps'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--grad-accum',
        type=int,
        default=16,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Base learning rate'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=200,
        help='Warmup steps'
    )
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )

    args = parser.parse_args()

    train(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        max_seq_len=args.max_seq_len,
        device=args.device,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()
