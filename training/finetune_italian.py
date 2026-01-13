"""
NanoPrime v2 - Phase 2: Italian Instruction Fine-Tuning 🇮🇹

Goal: Teach the model Italian while preserving its English reasoning capabilities.
Strategy:
1. Load pre-trained NanoPrime v2 (English)
2. Freeze lower layers (Syntax/Core) -> "Fixed Lower Layers"
3. Fine-tune higher layers (Semantics/Language) on Camoscio dataset
"""
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import NanoPrimeConfig
from src.model import NanoPrime
from data.italian_dataset import ItalianInstructDataset
from transformers import AutoTokenizer

def train_italian():
    # ===== CONFIGURATION =====
    config = NanoPrimeConfig()
    config.n_layers = 6
    config.d_model = 768
    config.batch_size = 16 # Adjustable
    config.max_seq_len = 128 # Match Cosmopedia Training
    config.use_router = False # Keep architecture consistent
    
    # Fine-tuning params
    TOTAL_STEPS = 5000  # Rapid adaptation
    SAVE_INTERVAL = 1000
    config.learning_rate = 5e-5  # Low LR for fine-tuning (gentle updates)
    config.weight_decay = 0.01
    
    print("=" * 70)
    print("🇮🇹 NanoPrime v2 - Italian Fine-Tuning (Phase 2)")
    print("=" * 70)
    
    # 1. Load Model
    model = NanoPrime(config).to(config.device)
    
    # Load Pre-trained Checkpoint
    checkpoint_path = 'nanoprime_cosmopedia_final.pth' # The output of Phase 1
    if os.path.exists(checkpoint_path):
        print(f"✓ Loading pre-trained base model: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=config.device)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"⚠️ Warning during load: {e}")
    else:
        print(f"❌ CRITICAL: Pre-trained checkpoint '{checkpoint_path}' not found!")
        print("   Please complete Phase 1 training (Cosmopedia) first.")
        return

    # 2. Freeze Lower Layers (Strategy: "Fixed Lower Layers")
    # We freeze the first 3 layers (embeddings + first 3 blocks)
    # effectively keeping the "core reasoning" stable.
    print("\n❄️ Freezing Strategy (Adapters approach):")
    
    # Freeze Embeddings
    for param in model.token_emb.parameters(): param.requires_grad = False
    for param in model.pos_emb.parameters(): param.requires_grad = False
    print("  - Embeddings: Frozen 🔒")
    
    # Freeze first N blocks
    N_FREEZE = 3
    for i in range(N_FREEZE):
        for param in model.blocks[i].parameters():
            param.requires_grad = False
        print(f"  - Block {i}: Frozen 🔒")
        
    print(f"  - Blocks {N_FREEZE}-{config.n_layers-1}: Trainable 🔥 (Italian Adaptation)")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable Parameters: {trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M ({trainable_params/total_params:.1%})\n")

    # 3. Dataset (Streaming)
    print("Connecting to Italian Data Stream...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = ItalianInstructDataset(tokenizer, max_length=config.max_seq_len, batch_size=config.batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0) # Batching handled by dataset
    
    # 4. Optimizer & Training Setup
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # Only optimize unfrozen
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == 'cuda'))
    model.train()
    
    print(f"=== Starting Fine-Tuning ({TOTAL_STEPS} steps) ===\n")
    pbar = tqdm(range(TOTAL_STEPS))
    data_iter = iter(dataloader)
    
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        x = batch[:, :-1].to(config.device)
        y = batch[:, 1:].to(config.device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=(config.device == 'cuda')):
            logits, loss = model(x, targets=y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_description(f"loss: {loss.item():.4f}")
        
        if (step + 1) % SAVE_INTERVAL == 0:
            save_path = f"nanoprime_ita_step_{step+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"\n✓ Saved checkpoint: {save_path}")
            
            # Simple Generation Test
            model.eval()
            prompt_txt = "User: Ciao, come stai?\nAI:"
            print(f"\n--- Test ({step+1}) ---")
            print(f"Input: {prompt_txt}")
            # (Simplified generation for log)
            # ... implementation skipped for brevity in log ...
            model.train()

    # Final Save
    torch.save(model.state_dict(), "nanoprime_v2_italian.pth")
    print("\n🇮🇹 Fine-Tuning Complete! Model saved to 'nanoprime_v2_italian.pth'")

if __name__ == "__main__":
    train_italian()
