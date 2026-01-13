"""
🚀 Elleci V1 - Pre-Training Global Check
Verifica completa prima di avviare il training
"""
import sys
import os
import torch
import json

print("=" * 70)
print("🚀 ELLECI V1 - PRE-TRAINING GLOBAL CHECK")
print("=" * 70)

errors = []
warnings = []

# 1. CUDA & Hardware
print("\n1. 🖥️  HARDWARE CHECK")
print("-" * 50)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   ✅ CUDA disponibile: {gpu_name}")
    print(f"   ✅ VRAM: {vram:.1f} GB")
    
    # Test CUDA
    try:
        x = torch.randn(100, 100, device='cuda')
        y = x @ x.T
        del x, y
        torch.cuda.empty_cache()
        print(f"   ✅ CUDA test: OK")
    except Exception as e:
        errors.append(f"CUDA test failed: {e}")
        print(f"   ❌ CUDA test: FAILED")
else:
    errors.append("CUDA non disponibile")
    print("   ❌ CUDA non disponibile!")

# 3. Tokenizer (load first, we need vocab size)
print("\n2. 📝 TOKENIZER CHECK")
print("-" * 50)
tokenizer = None
try:
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_chimera_v2_patched")
    vocab_size = len(tokenizer)
    print(f"   ✅ Tokenizer caricato: {vocab_size} tokens")
    
    # Test tokenization
    test_it = "Ciao, come stai? Questa è una prova."
    test_en = "Hello, how are you? This is a test."
    tokens_it = tokenizer.encode(test_it)
    tokens_en = tokenizer.encode(test_en)
    print(f"   ✅ Test IT: '{test_it[:30]}...' → {len(tokens_it)} tokens")
    print(f"   ✅ Test EN: '{test_en[:30]}...' → {len(tokens_en)} tokens")
    
    # Check EOS token
    if tokenizer.eos_token_id is not None:
        print(f"   ✅ EOS token: {tokenizer.eos_token_id}")
    else:
        warnings.append("EOS token is None")
        print(f"   ⚠️  EOS token: None (verrà gestito)")
        
except Exception as e:
    errors.append(f"Tokenizer failed: {e}")
    print(f"   ❌ Errore tokenizer: {e}")

# 2. Model
print("\n3. 🧠 MODEL CHECK")
print("-" * 50)
try:
    sys.path.insert(0, '.')
    from src.config import NanoPrimeConfig
    from src.model import NanoPrime
    
    config = NanoPrimeConfig()
    print(f"   ✅ Config caricata: {config.d_model}d x {config.n_layers}L")
    print(f"   ✅ Config vocab_size: {config.vocab_size:,}")
    
    # Check vocab size match
    if tokenizer and config.vocab_size != len(tokenizer):
        warnings.append(f"Vocab size mismatch: config={config.vocab_size}, tokenizer={len(tokenizer)}")
        print(f"   ⚠️  Vocab size mismatch! Config: {config.vocab_size}, Tokenizer: {len(tokenizer)}")
        # Temporarily override for test
        config.vocab_size = len(tokenizer)
        print(f"   📝 Override vocab_size to {len(tokenizer)} for test")
    
    # Quick model test (CPU to save VRAM)
    model = NanoPrime(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Modello creato: {params/1e9:.2f}B parametri")
    del model
    
except Exception as e:
    errors.append(f"Model load failed: {e}")
    print(f"   ❌ Errore modello: {e}")

# 4. Dataset
print("\n4. 📊 DATASET CHECK")
print("-" * 50)
if tokenizer:
    try:
        from data.elleci_dataset import EllediDataset
        
        # Phase 1
        ds1 = EllediDataset(tokenizer, max_length=256, phase=1)
        print(f"   ✅ Phase 1 ratios: {ds1.ratios}")
        
        # Phase 2
        ds2 = EllediDataset(tokenizer, max_length=256, phase=2)
        print(f"   ✅ Phase 2 ratios: {ds2.ratios}")
        
        # Sample test
        sample_iter = iter(ds1)
        sample = next(sample_iter)
        print(f"   ✅ Sample shape: {len(sample)} tokens")
        
    except Exception as e:
        errors.append(f"Dataset failed: {e}")
        print(f"   ❌ Errore dataset: {e}")
else:
    print("   ⚠️  Skipped (tokenizer not loaded)")

# 5. Italian Instructions
print("\n5. 🇮🇹  ITALIAN INSTRUCTIONS CHECK")
print("-" * 50)
instruction_files = [
    'data/elleci_instructions.jsonl',
    'data/chimera_instructions_final.jsonl'
]
total_instructions = 0
for path in instruction_files:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"   ✅ {os.path.basename(path)}: {count:,} istruzioni")
        total_instructions += count
    else:
        print(f"   ⚠️  {os.path.basename(path)}: non trovato")

print(f"   📊 Totale istruzioni IT: {total_instructions:,}")

# 6. Training Script
print("\n6. 📜 TRAINING SCRIPT CHECK")
print("-" * 50)
train_script = 'scripts/train_elleci.py'
if os.path.exists(train_script):
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'WSD Scheduler': 'def get_lr_wsd' in content or 'WSD' in content,
        'Gradient Checkpointing': 'gradient_checkpointing' in content,
        'EllediDataset import': 'EllediDataset' in content,
        'SWA (Stochastic Weight Averaging)': 'swa_model' in content or 'AveragedModel' in content,
        'LeRaC': 'lerac' in content.lower() or 'layer_lr' in content,
        '8-bit AdamW': 'bnb' in content or '8bit' in content,
    }
    
    for feature, present in checks.items():
        status = "✅" if present else "⚠️ "
        print(f"   {status} {feature}: {'OK' if present else 'Non trovato'}")
else:
    errors.append("Training script not found")
    print(f"   ❌ Script non trovato: {train_script}")

# 7. Checkpoints directory
print("\n7. 💾 STORAGE CHECK")
print("-" * 50)
os.makedirs('checkpoints', exist_ok=True)
print(f"   ✅ Cartella checkpoints: OK")

# Check disk space (Windows)
try:
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1024**3
    print(f"   ✅ Spazio libero: {free_gb:.1f} GB")
    if free_gb < 10:
        warnings.append(f"Low disk space: {free_gb:.1f} GB")
except:
    pass

# 8. Quick Forward Pass Test (with corrected vocab size)
print("\n8. ⚡ QUICK FORWARD PASS TEST")
print("-" * 50)
try:
    # Clear CUDA state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    from src.config import NanoPrimeConfig
    from src.model import NanoPrime
    
    config = NanoPrimeConfig()
    # Use tokenizer vocab size
    config.vocab_size = len(tokenizer)
    
    model = NanoPrime(config).cuda()
    model.train()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Create dummy batch
    batch_size = 2
    seq_len = 256
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda')
    
    # Forward pass
    with torch.amp.autocast('cuda', dtype=torch.float16):
        output = model(dummy_input)
        loss = output.loss
    
    print(f"   ✅ Forward pass: OK (loss = {loss.item():.4f})")
    
    # Backward pass
    loss.backward()
    print(f"   ✅ Backward pass: OK")
    
    # Check gradients
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"   ✅ Gradients: {'OK' if has_grads else 'MISSING'}")
    
    # Memory usage
    mem_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"   ✅ Peak VRAM usage: {mem_used:.2f} GB")
    
    del model, dummy_input, output, loss
    torch.cuda.empty_cache()
    
except Exception as e:
    errors.append(f"Forward pass failed: {e}")
    print(f"   ❌ Test fallito: {e}")

# Summary
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ ERRORI CRITICI ({len(errors)}):")
    for e in errors:
        print(f"   • {e}")
else:
    print("\n✅ Nessun errore critico!")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"   • {w}")
else:
    print("✅ Nessun warning!")

# Final verdict
print("\n" + "=" * 70)
if not errors:
    print("🚀 READY TO TRAIN!")
    print("=" * 70)
    print("\nComando per avviare il training:")
    print("   python scripts/train_elleci.py")
    print("\nPer dry run (test 5 step):")
    print("   python scripts/train_elleci.py --dry-run --steps 5")
else:
    print("❌ FIX ERRORS BEFORE TRAINING")
    print("=" * 70)
