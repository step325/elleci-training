"""
🚀 Elleci V1 - COMPREHENSIVE Pre-Training Check
Verifies ALL components before starting training
"""
import sys
import os
import torch
import time

print("=" * 70)
print("🚀 ELLECI V1 - COMPREHENSIVE PRE-TRAINING CHECK")
print("=" * 70)

errors = []
warnings = []

# ============================================================
# 1. CUDA & Hardware
# ============================================================
print("\n1. 🖥️  HARDWARE CHECK")
print("-" * 50)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   ✅ CUDA: {gpu_name}")
    print(f"   ✅ VRAM: {vram:.1f} GB")
    
    # CUDA test
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    del x, y
    torch.cuda.empty_cache()
    print(f"   ✅ CUDA matmul: OK")
else:
    errors.append("CUDA not available")
    print("   ❌ CUDA not available!")

# ============================================================
# 2. Tokenizer
# ============================================================
print("\n2. 📝 TOKENIZER CHECK")
print("-" * 50)
tokenizer = None
try:
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_chimera_v2_patched")
    vocab_size = len(tokenizer)
    print(f"   ✅ Loaded: {vocab_size:,} tokens")
    
    test_it = "Ciao, come stai?"
    test_en = "Hello, how are you?"
    print(f"   ✅ IT: {len(tokenizer.encode(test_it))} tokens")
    print(f"   ✅ EN: {len(tokenizer.encode(test_en))} tokens")
    print(f"   ✅ EOS: {tokenizer.eos_token_id}")
except Exception as e:
    errors.append(f"Tokenizer: {e}")
    print(f"   ❌ Error: {e}")

# ============================================================
# 3. Model with Mamba-2
# ============================================================
print("\n3. 🧠 MODEL + MAMBA-2 CHECK")
print("-" * 50)
try:
    sys.path.insert(0, '.')
    from src.config import NanoPrimeConfig
    from src.model import NanoPrime
    from src.modules.mamba2 import Mamba2BlockFast
    
    config = NanoPrimeConfig()
    config.vocab_size = len(tokenizer) if tokenizer else 32000
    config.mamba.use_mamba2 = True
    
    print(f"   ✅ Config: {config.d_model}d x {config.n_layers}L")
    print(f"   ✅ Mamba-2 enabled: {config.mamba.use_mamba2}")
    print(f"   ✅ Mamba n_heads: {config.mamba.n_heads}")
    print(f"   ✅ Mamba d_state: {config.mamba.d_state}")
    print(f"   ✅ Mamba chunk_size: {config.mamba.chunk_size}")
    
    # Create model
    model = NanoPrime(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model: {params/1e6:.1f}M params")
    
    # Verify Mamba-2 blocks are being used
    mamba2_count = 0
    mamba1_count = 0
    for name, module in model.named_modules():
        if 'Mamba2BlockFast' in str(type(module)):
            mamba2_count += 1
        elif 'MambaBlock' in str(type(module)) and 'Mamba2' not in str(type(module)):
            mamba1_count += 1
    
    print(f"   ✅ Mamba-2 blocks: {mamba2_count}")
    print(f"   ⚠️  Mamba-1 blocks: {mamba1_count}")
    
    if mamba2_count == 0 and mamba1_count > 0:
        warnings.append("Mamba-1 blocks found instead of Mamba-2")
    
    del model
    
except Exception as e:
    errors.append(f"Model: {e}")
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 4. Mamba-2 Speed Test
# ============================================================
print("\n4. ⚡ MAMBA-2 SPEED TEST")
print("-" * 50)
try:
    from src.modules.mamba2 import Mamba2Block, Mamba2BlockFast
    from src.modules.mamba import MambaBlock
    from dataclasses import dataclass
    
    @dataclass
    class TestMambaConfig:
        d_model: int = 768
        d_state: int = 16
        d_conv: int = 4
        expand: int = 2
        n_heads: int = 8
        chunk_size: int = 64
        dt_rank: int = 48
    
    test_config = TestMambaConfig()
    
    # Create blocks
    mamba1 = MambaBlock(test_config).cuda()
    mamba2_fast = Mamba2BlockFast(test_config).cuda()
    
    x_test = torch.randn(4, 256, 768, device='cuda')
    
    # Warmup
    for _ in range(3):
        _ = mamba1(x_test)
        _ = mamba2_fast(x_test)
    torch.cuda.synchronize()
    
    # Benchmark Mamba-1
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = mamba1(x_test)
    torch.cuda.synchronize()
    time_v1 = (time.time() - start) * 100  # ms per iter
    
    # Benchmark Mamba-2 Fast
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = mamba2_fast(x_test)
    torch.cuda.synchronize()
    time_v2 = (time.time() - start) * 100  # ms per iter
    
    speedup = time_v1 / time_v2
    print(f"   ✅ Mamba-1: {time_v1:.1f}ms")
    print(f"   ✅ Mamba-2 Fast: {time_v2:.1f}ms")
    print(f"   ✅ Speedup: {speedup:.2f}x")
    
    if speedup < 1.5:
        warnings.append(f"Mamba-2 speedup lower than expected: {speedup:.2f}x")
    
    del mamba1, mamba2_fast, x_test
    torch.cuda.empty_cache()
    
except Exception as e:
    errors.append(f"Mamba-2 speed test: {e}")
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 5. Dataset
# ============================================================
print("\n5. 📊 DATASET CHECK")
print("-" * 50)
if tokenizer:
    try:
        from data.elleci_dataset import EllediDataset
        
        ds1 = EllediDataset(tokenizer, max_length=256, phase=1)
        print(f"   ✅ Phase 1: {ds1.ratios}")
        
        ds2 = EllediDataset(tokenizer, max_length=256, phase=2)
        print(f"   ✅ Phase 2: {ds2.ratios}")
        
        # Sample
        sample_iter = iter(ds1)
        sample = next(sample_iter)
        print(f"   ✅ Sample: {len(sample)} tokens")
        
    except Exception as e:
        errors.append(f"Dataset: {e}")
        print(f"   ❌ Error: {e}")

# ============================================================
# 6. Italian Instructions
# ============================================================
print("\n6. 🇮🇹  INSTRUCTIONS CHECK")
print("-" * 50)
total = 0
for path in ['data/elleci_instructions.jsonl', 'data/chimera_instructions_final.jsonl']:
    if os.path.exists(path):
        count = sum(1 for _ in open(path, encoding='utf-8'))
        print(f"   ✅ {os.path.basename(path)}: {count:,}")
        total += count
print(f"   📊 Total: {total:,} instructions")

# ============================================================
# 7. Training Script Features
# ============================================================
print("\n7. 📜 TRAINING FEATURES CHECK")
print("-" * 50)
script = 'scripts/train_elleci.py'
if os.path.exists(script):
    content = open(script, encoding='utf-8').read()
    features = {
        'Mamba-2 enabled': 'use_mamba2 = True' in content,
        'WSD Scheduler': 'WSD' in content or 'wsd' in content.lower(),
        'Gradient Checkpointing': 'gradient_checkpointing' in content,
        'EllediDataset': 'EllediDataset' in content,
        'SWA': 'swa_model' in content or 'AveragedModel' in content,
        'LeRaC': 'lerac' in content.lower() or 'layer_lr' in content,
        '8-bit AdamW': 'bnb.' in content or 'bitsandbytes' in content,
    }
    for f, ok in features.items():
        print(f"   {'✅' if ok else '❌'} {f}")
        if not ok:
            warnings.append(f"Feature not found: {f}")

# ============================================================
# 8. Forward + Backward Pass Test
# ============================================================
print("\n8. 🔥 FORWARD + BACKWARD PASS TEST")
print("-" * 50)
try:
    # Reset CUDA state completely
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    config = NanoPrimeConfig()
    # CRITICAL: Set vocab_size BEFORE creating model
    actual_vocab_size = len(tokenizer) if tokenizer else 32043
    config.vocab_size = actual_vocab_size
    config.mamba.use_mamba2 = True
    
    print(f"   📊 Using vocab_size={config.vocab_size}")
    
    model = NanoPrime(config).cuda()
    model.train()
    model.gradient_checkpointing_enable()
    
    batch_size = 4
    seq_len = 256
    # Generate tokens WITHIN vocab bounds (0 to vocab_size-1)
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda')
    print(f"   📊 Token range: [{x.min().item()}, {x.max().item()}] (max allowed: {config.vocab_size-1})")
    
    # Forward
    torch.cuda.synchronize()
    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = model(x)
        loss = out.loss if hasattr(out, 'loss') else out[1] if isinstance(out, tuple) else None
    torch.cuda.synchronize()
    fwd_time = time.time() - start
    
    if loss is None:
        # Manual loss
        logits = out.logits if hasattr(out, 'logits') else out[0]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1)
        )
    
    print(f"   ✅ Forward: {fwd_time*1000:.1f}ms, Loss={loss.item():.4f}")
    
    # Backward
    start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = time.time() - start
    
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"   ✅ Backward: {bwd_time*1000:.1f}ms")
    print(f"   ✅ Gradients: {has_grads}/{total_params} params")
    
    # Memory
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"   ✅ Peak VRAM: {peak_mem:.2f} GB")
    
    del model, x, out, loss
    torch.cuda.empty_cache()
    
except Exception as e:
    errors.append(f"Forward/Backward: {e}")
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 9. Disk Space
# ============================================================
print("\n9. 💾 STORAGE CHECK")
print("-" * 50)
import shutil
os.makedirs('checkpoints', exist_ok=True)
total, used, free = shutil.disk_usage(".")
print(f"   ✅ Free space: {free/1024**3:.1f} GB")
if free/1024**3 < 10:
    warnings.append("Low disk space!")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ CRITICAL ERRORS ({len(errors)}):")
    for e in errors:
        print(f"   • {e}")
else:
    print("\n✅ No critical errors!")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"   • {w}")
else:
    print("✅ No warnings!")

print("\n" + "=" * 70)
if not errors:
    print("🚀 ALL SYSTEMS GO - READY FOR TRAINING!")
    print("=" * 70)
    print("\nCommand to start training:")
    print("   python scripts/train_elleci.py")
else:
    print("❌ FIX ERRORS BEFORE TRAINING")
    print("=" * 70)
