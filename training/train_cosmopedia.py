"""
NanoPrime v2 - Production Training on Cosmopedia

Phase 1 of roadmap: Real dataset for quality improvement.

Configuration:
- Dataset: Cosmopedia (100K educational texts)
- Steps: 50,000 (vs 10K before)
- Batch: 16 (effective ~1000 tokens/batch)
- Max seq: 256 (vs 64 - real content is longer)
- ETA: ~3-4 hours
"""
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import NanoPrimeConfig
from src.model import NanoPrime
from data.cosmopedia import CosmopediaDataset


def train_cosmopedia():
    """Train on real educational content"""
    
    # ===== CONFIGURATION =====
    config = NanoPrimeConfig()
    config.n_layers = 6  # Keep same architecture as v2
    config.d_model = 768
    config.batch_size = 32 # Base batch size (Will be 64 total on 2 GPUs)
    config.max_seq_len = 128  # Reduced from 256 for speed (4-5x faster!)
    config.use_router = False # <--- CRITICAL: Disable router for speed (2x faster) & stability
    
    # Training params
    TOTAL_STEPS = 15000  # Optimized for 12h limit (960k samples total)
    SAVE_INTERVAL = 2500 # Save every ~2 hours
    config.learning_rate = 3e-4  # Lowered from 1e-3 for stability
    config.weight_decay = 0.01
    config.max_grad_norm = 1.0
    
    # Logging
    LOG_INTERVAL = 100
    EVAL_INTERVAL = 500
    
    print("=" * 70)
    print("NanoPrime v2 - Cosmopedia Training (Production)")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Architecture: BitNet + Mamba + MLA (v2 frozen)")
    print(f"Dataset: Cosmopedia (100K educational texts)")
    print(f"Batch: {config.batch_size} × {config.max_seq_len} = {config.batch_size * config.max_seq_len} tokens")
    print(f"Total steps: {TOTAL_STEPS:,}")
    print(f"Estimated time: ~10-12 hours (optimized!)")
    print()
    
    # Create model
    model = NanoPrime(config).to(config.device)
    
    # Multi-GPU Support
    if torch.cuda.device_count() > 1:
        print(f"✓ Found {torch.cuda.device_count()} GPUs! Using DataParallel.")
        model = nn.DataParallel(model)
        # Scale batch size effectively
        config.batch_size *= torch.cuda.device_count() 
        print(f"✓ Scaled batch size to {config.batch_size} (Global)")
    
    # Standardize & Compile for speed
    torch.backends.cudnn.benchmark = True
    
    # Check GPU capability for torch.compile (Requires Compute Capability >= 7.0)
    if hasattr(torch, "compile") and config.device == "cuda":
        gpu_cap = torch.cuda.get_device_capability()
        if gpu_cap[0] >= 7:
            print(f"✓ GPU Capability {gpu_cap[0]}.{gpu_cap[1]} >= 7.0: Enabling torch.compile()...")
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}. Continuing without compilation.")
        else:
            print(f"⚠️ GPU Capability {gpu_cap[0]}.{gpu_cap[1]} < 7.0 (e.g. P100). Skipping torch.compile.")
    
    # Load v2 checkpoint as starting point (optional)
    try:
        v2_checkpoint = torch.load('nanoprime_v2_final.pth', map_location=config.device)
        model.load_state_dict(v2_checkpoint)
        print("✓ Loaded v2 checkpoint as initialization")
    except:
        print("⚠️ Starting from scratch (v2 checkpoint not found)")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model ready: {total_params/1e6:.1f}M parameters\n")
    
    # Load Cosmopedia dataset (STREAMING - instant start!)
    try:
        dataset = CosmopediaDataset(
            split='train',
            max_length=config.max_seq_len,
            subset='web_samples_v2'  # Streaming mode, no n_samples needed!
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=2,         # Enable parallel downloading!
            prefetch_factor=2,     # Buffer 2 batches per worker
            pin_memory=True,
            persistent_workers=True # Keep workers alive
        )
        print("✓ Cosmopedia streaming dataset ready")
        print("  (Training starts IMMEDIATELY!)\n")
    except Exception as e:
        print(f"❌ Failed to load Cosmopedia: {e}")
        print("Please install: pip install datasets")
        return
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Custom schedule with Warmup
    def get_lr_schedule(step):
        warmup_steps = 1000
        if step < warmup_steps:
             return float(step) / float(max(1, warmup_steps))
        # Cosine decay after warmup
        progress = float(step - warmup_steps) / float(max(1, TOTAL_STEPS - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_schedule)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(config.device == 'cuda'))
    
    # Create dirs
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Training loop
    model.train()
    data_iter = iter(dataloader)
    
    print(f"=== Training Start ({TOTAL_STEPS:,} steps) ===\n")
    
    pbar = tqdm(range(TOTAL_STEPS), desc="Training")
    best_loss = float('inf')
    
    for step in pbar:
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        
        x, y = x.to(config.device), y.to(config.device)
        
        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        
        # Use float16 for P100/T4 compatibility (BF16 requires Ampere+)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(config.device == 'cuda')):
            logits, loss = model(x, targets=y)
            
            # CRITICAL: Handle multi-GPU loss (it returns a vector)
            if loss.ndim > 0:
                loss = loss.mean()
        
        # NaN check
        if torch.isnan(loss):
            print(f"❌ NaN detected at step {step}!")
            print(f"  LR: {optimizer.param_groups[0]['lr']}")
            breakpoint() # Or break
            break

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
        if (step + 1) % LOG_INTERVAL == 0:
            print(f"\nStep {step+1:,}/{TOTAL_STEPS:,}: Loss={loss.item():.4f}, LR={current_lr:.6f}")
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                print(f"  ✓ New best loss: {best_loss:.4f}")
        
        # Validation
        if (step + 1) % EVAL_INTERVAL == 0:
            print(f"\n{'='*70}")
            print(f"Validation at step {step+1:,}")
            print(f"{'='*70}")
            
            model.eval()
            with torch.no_grad():
                # Generate sample (Handle DataParallel wrapper)
                prompt = torch.zeros((1, 1), dtype=torch.long, device=config.device)
                
                # Unwrap model for generation
                raw_model = model.module if hasattr(model, "module") else model
                generated = raw_model.generate(prompt, max_new_tokens=50, temperature=0.8)
                
                # Decode
                text = dataset.tokenizer.decode(generated[0].tolist())
                print(f"Generated: {text[:200]}...")
            
            model.train()
            print(f"{'='*70}\n")
        
        # Save checkpoint (Unwrap for compatibility)
        if (step + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = f"checkpoints/nanoprime_cosmopedia_step_{step+1}.pth"
            raw_model = model.module if hasattr(model, "module") else model
            torch.save(raw_model.state_dict(), checkpoint_path)
            print(f"✓ Saved: {checkpoint_path}")
    
    # Final save
    print("\n=== Training Complete ===")
    final_path = "nanoprime_cosmopedia_final.pth"
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), final_path)
    print(f"✓ Saved final model: {final_path}")
    print(f"✓ Best loss achieved: {best_loss:.4f}")
    
    print("\n🎉 Cosmopedia training finished!")
    print(f"Next: Test with inference_v2.py --checkpoint {final_path}")


if __name__ == "__main__":
    train_cosmopedia()
