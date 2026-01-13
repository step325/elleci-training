"""
NanoPrime v2.0 - Training Script

Main training loop with:
- Checkpointing
- Logging
- Validation
- Mixed precision
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import NanoPrimeConfig
from src.model import NanoPrime
from data.tinystories import TinyStoriesDataset, InfiniteTinyStories


def train():
    """Main training function"""
    
    # Configuration (OPTIMIZED)
    config = NanoPrimeConfig()
    config.n_layers = 6  # 6 layers for good balance
    config.batch_size = 16  # Increased from 8 (more examples per step)
    config.max_seq_len = 64  # Reduced from 256 (TinyStories are short ~13 tokens)
    
    print("=" * 70)
    print("NanoPrime v2.0 Training")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Layers: {config.n_layers}")
    print(f"d_model: {config.d_model}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max seq len: {config.max_seq_len}")
    print()
    
    # Create model
    model = NanoPrime(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params/1e6:.1f}M parameters\n")
    
    # Dataset
    dataset = InfiniteTinyStories(max_length=config.max_seq_len, vocab_size=config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    print("✓ Dataset ready\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TOTAL_STEPS = 10000
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1e-5)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(config.device == 'cuda' and config.mixed_precision == 'bf16'))
    
    # Training loop
    model.train()
    data_iter = iter(dataloader)
    
    print(f"=== Training Start ({TOTAL_STEPS} steps) ===\n")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    pbar = tqdm(range(TOTAL_STEPS), desc="Training")
    
    for step in pbar:
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        
        x, y = x.to(config.device), y.to(config.device)
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(config.device == 'cuda')):
            logits, loss = model(x, targets=y, use_router=False)  # Router off for now
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update progress
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_description(f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
        # Logging
        if (step + 1) % config.log_interval == 0:
            print(f"\nStep {step+1}: Loss={loss.item():.4f}, LR={current_lr:.6f}")
        
        # Validation
        if (step + 1) % config.eval_interval == 0:
            print(f"\n{'='*70}")
            print(f"Validation at step {step+1}")
            print(f"{'='*70}")
            
            # Simple generation test
            model.eval()
            with torch.no_grad():
                prompt = torch.zeros((1, 1), dtype=torch.long, device=config.device)
                generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
                print(f"Generated tokens: {generated[0].tolist()[:30]}")
            model.train()
            
            print(f"{'='*70}\n")
        
        # Save checkpoint
        if (step + 1) % config.save_interval == 0:
            checkpoint_path = f"checkpoints/nanoprime_step_{step+1}.pth"
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'config': config.to_dict(),
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Final save
    print("\n=== Training Complete ===")
    final_path = "nanoprime_v2_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"✓ Saved final model: {final_path}")
    
    print("\n🎉 Training finished successfully!")


if __name__ == "__main__":
    train()
