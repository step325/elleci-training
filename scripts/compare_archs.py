
"""
Architecture Comparison Script (A/B Testing)

Compares 3 variants of NanoPrime V2:
1. Baseline: GELU + LayerNorm (Current)
2. V2-A: SwiGLU + RMSNorm (Performance focus)
3. V2-B: SquaredReLU + RMSNorm (Quantization stability focus)

Metrics:
- Loss Curve
- Gradient Norm (spike detection)
- Throughput (speed)
- Activation Stats (std dev, gate sparsity)
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
# import pandas as pd # Unused
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# Add root to path
sys.path.append(os.getcwd())

from src.config import NanoPrimeConfig
from src.model_v2 import NanoPrimeV2
from data.chimera_dataset import ChimeraDataset
from transformers import PreTrainedTokenizerFast

# Configuration
STEPS = 200
BATCH_SIZE = 4
SEQ_LEN = 512  # Fixed sequence length for comparison
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_model(variant):
    """Initialize model based on variant name"""
    config = NanoPrimeConfig(
        vocab_size=32128,
        max_seq_len=SEQ_LEN,
        d_model=768, # Smaller model for fast testing
        n_layers=6,
        use_router=False
    )
    
    if variant == "Baseline":
        config.ffn_type = "gelu"
        config.norm_type = "layer"
    elif variant == "V2-A (SwiGLU)":
        config.ffn_type = "swiglu"
        config.norm_type = "rms"
    elif variant == "V2-B (SqReLU)":
        config.ffn_type = "sqrelu"
        config.norm_type = "rms"
        
    model = NanoPrimeV2(config).to(config.device)
    return model, config

def train_variant(variant, dataloader, device):
    print(f"\n🧪 Testing Variant: {variant}")
    set_seed(SEED) # Reset seed for fair comparison
    
    model, config = get_model(variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        "loss": [],
        "grad_norm": [],
        "throughput": [],
        "activation_std": [] # Placeholder
    }
    
    model.train()
    start_time = time.time()
    
    # Hook to capture activation stats (optional, simplified for now)
    # We can add hooks if needed, but for now we look at loss stability
    
    pbar = tqdm(range(STEPS), desc=variant)
    iter_loader = iter(dataloader)
    
    for step in pbar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)
            
        input_ids = batch['input_ids'].to(device)
        targets = batch['labels'].to(device)
        
        # Forward
        t0 = time.time()
        logits = model(input_ids)
        
        # Reshape for loss
        B, T, V = logits.shape
        loss = criterion(logits.view(-1, V), targets.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        t1 = time.time()
        
        # Metrics
        metrics["loss"].append(loss.item())
        metrics["grad_norm"].append(grad_norm.item())
        metrics["throughput"].append(B * T / (t1 - t0))
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "gnorm": f"{grad_norm.item():.2f}"})
        
    total_time = time.time() - start_time
    print(f"✅ {variant} finished in {total_time:.2f}s")
    
    return metrics

def main():
    print("🚀 Starting A/B Comparison...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Use Synthetic Data for Speed/Reliability
    print("⚠️ Using Synthetic Data for Architecture Comparison")
    
    class SyntheticDataset(IterableDataset):
        def __iter__(self):
            while True:
                yield {
                    'input_ids': torch.randint(0, 32128, (SEQ_LEN,)),
                    'labels': torch.randint(0, 32128, (SEQ_LEN,))
                }
    
    dataset = SyntheticDataset()
    print(f"Dataset ready (Synthetic)")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    variants = ["Baseline", "V2-A (SwiGLU)", "V2-B (SqReLU)"]
    results = {}
    
    for v in variants:
        results[v] = train_variant(v, dataloader, device)
        
    # Analysis
    print("\n📊 Final Results Summary (Avg over last 50 steps)")
    print(f"{'Variant':<20} | {'Loss':<10} | {'GradNorm':<10} | {'Tokens/Sec':<10}")
    print("-" * 60)
    
    for v, m in results.items():
        avg_loss = np.mean(m["loss"][-50:])
        avg_gnorm = np.mean(m["grad_norm"][-50:])
        avg_speed = np.mean(m["throughput"])
        
        print(f"{v:<20} | {avg_loss:<10.4f} | {avg_gnorm:<10.2f} | {avg_speed:<10.0f}")

    # Check for instability
    print("\n⚠️ Instability Check:")
    for v, m in results.items():
        max_gnorm = np.max(m["grad_norm"])
        if max_gnorm > 10.0: # Arbitrary threshold for spike
             print(f"FAILED: {v} had gradient spike (max: {max_gnorm:.2f})")
        else:
             print(f"PASSED: {v} (max gnorm: {max_gnorm:.2f})")

if __name__ == "__main__":
    main()
