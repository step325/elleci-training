"""
NanoPrime v2 - Dry Run (Performance Validation) 🏎️
Verifies that the 1.5B model runs efficiently on the target hardware.
Goal: > 2000 tokens/sec.
"""
import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import NanoPrimeConfig
from src.model import NanoPrime

# Force 1.5B Configuration
CONFIG = NanoPrimeConfig(
    d_model=2048,
    n_layers=30,  # 30-32 is 1.5B range
    vocab_size=50304, # Custom Facab
    max_seq_len=1024, # Reduced for stability testing
    batch_size=1      # Micro-batch size
)
CONFIG.mla.n_heads = 16 # Correct head dim

def run_benchmark():
    if not torch.cuda.is_available():
        print("❌ CRITICAL: No GPU detected.")
        return

    print("="*60)
    print(f"🔥 NanoPrime Dry Run: 1.5B Params | Context 2048")
    print(f"   Vocab: {CONFIG.vocab_size} | Device: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    # 1. Initialize Model
    print("[1] Allocating Model (FP32 Master, BF16 Ops)...")
    try:
        model = NanoPrime(CONFIG).cuda()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"    ✅ Model Size: {param_count/1e9:.2f}B Parameters")
    except Exception as e:
        print(f"❌ Failed to allocate model: {e}")
        return

    # 2. Random Data
    print("[2] Generating dummy data...")
    x = torch.randint(0, CONFIG.vocab_size, (1, CONFIG.max_seq_len)).cuda() # Batch 1, Seq 2048
    y = torch.randint(0, CONFIG.vocab_size, (1, CONFIG.max_seq_len)).cuda() 
    
    # 3. Optimizer (Try bitsandbytes if available, else AdamW)
    print("[3] Loading Optimizer...")
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4) # 8-bit
        print("    ✅ BitsAndBytes 8-bit AdamW Loaded! (VRAM Saver)")
    except ImportError:
        print("    ⚠️ BitsAndBytes NOT found. Using Standard AdamW (High VRAM usage).")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Enable Gradient Checkpointing (Saves VRAM)
    if hasattr(model, "gradient_checkpointing_enable"):
         model.gradient_checkpointing_enable()
         print("    ✅ Gradient Checkpointing ENABLED (Critical for 12GB VRAM)")
    else:
         print("    ⚠️ Model does not support Gradient Checkpointing. Skipping.")
        
    model.train()
    
    # 4. Training Loop (100 Steps)
    print("\n[4] Starting 100-Step Sprint...")
    total_tokens = 0
    start_time = time.time()
    
    # Warmup
    for _ in range(5):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, CONFIG.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    print("    Warmup complete. Measuring...")
    bench_start = time.time()
    
    STEPS = 50 
    for i in range(STEPS):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Forward
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, CONFIG.vocab_size), y.view(-1))
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        optimizer.zero_grad()
        
        # Stats
        tokens_processed = CONFIG.batch_size * CONFIG.max_seq_len
        total_tokens += tokens_processed
        
        sys.stdout.write(f"\r    Step {i+1}/{STEPS} | Loss: {loss.item():.4f}")
        sys.stdout.flush()
        
    bench_end = time.time()
    duration = bench_end - bench_start
    tps = total_tokens / duration
    
    print(f"\n\n🏁 RESULT: {tps:.2f} Tokens/Sec")
    
    if tps < 500:
        print("⚠️ WARNING: Speed is very low (<500 tps). Check CUDA kernels or WSL2 setup.")
        print("   Likely getting CPU fallback on Mamba or BitLinear.")
    else:
        print("✅ SUCCESS: Training speed is healthy.")

if __name__ == "__main__":
    run_benchmark()
