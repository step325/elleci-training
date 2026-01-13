"""
Quick validation script to test if everything is set up correctly.
"""
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("NanoPrime v2.0 - Setup Validation")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from src.config import NanoPrimeConfig
    from src.model import NanoPrime
    from data.tinystories import TinyStoriesDataset
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test config
print("\n2. Testing configuration...")
try:
    config = NanoPrimeConfig()
    print(f"   ✓ Config created (d_model={config.d_model}, {config.n_layers} layers)")
except Exception as e:
    print(f"   ❌ Config error: {e}")
    sys.exit(1)

# Test model creation
print("\n3. Testing model creation...")
try:
    import torch
    config.n_layers = 2  # Small for testing
    model = NanoPrime(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created ({params/1e6:.1f}M parameters)")
except Exception as e:
    print(f"   ❌ Model creation error: {e}")
    sys.exit(1)

# Test forward pass
print("\n4. Testing forward pass...")
try:
    idx = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx, targets=idx)
    print(f"   ✓ Forward pass successful (loss={loss.item():.4f})")
except Exception as e:
    print(f"   ❌ Forward pass error: {e}")
    sys.exit(1)

# Test dataset
print("\n5. Testing dataset...")
try:
    dataset = TinyStoriesDataset(max_length=64, num_samples=10)
    x, y = dataset[0]
    print(f"   ✓ Dataset works ({len(dataset)} samples, shape={x.shape})")
except Exception as e:
    print(f"   ❌ Dataset error: {e}")
    sys.exit(1)

# Check CUDA
print("\n6. Checking CUDA...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   ✓ VRAM: {vram:.1f} GB")
else:
    print("   ⚠️  CUDA not available - will use CPU")

print("\n" + "=" * 60)
print("✅ All validation checks passed!")
print("=" * 60)
print("\nReady to train! Run:")
print("  python training/train.py")
print()
