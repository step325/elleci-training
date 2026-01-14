"""
Quick test for ChimeraDataset streaming.
Tests only the instructions (local, fast) to verify basic functionality.
"""
import sys
import os
from transformers import PreTrainedTokenizerFast

sys.path.append(os.getcwd())

from data.chimera_dataset import ChimeraDataset

def test_dataset_quick():
    """Quick test using only local instructions (no HuggingFace download)."""
    print("🔍 Quick Test - Local Instructions Only")
    print("="*60)

    print("\n[1/3] Loading Tokenizer...")
    tokenizer_path = "tokenizer_chimera_v2_patched/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        return False

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"

    vocab_size = len(tokenizer.get_vocab())
    print(f"✅ Tokenizer loaded (vocab_size={vocab_size})")

    print("\n[2/3] Creating ChimeraDataset (Phase 2 - 55% instructions)...")
    # Use Phase 2 which has 55% instructions for faster testing
    dataset = ChimeraDataset(
        tokenizer,
        phase=2,  # Phase 2 has more instructions
        max_length=256,
        batch_size=4
    )

    print("\n[3/3] Testing streaming (fetching 5 batches)...")
    print("⏳ This will download from HuggingFace on first run (may take 1-2 min)...")
    it = iter(dataset)

    success_count = 0
    for i in range(5):
        try:
            print(f"\n  📦 Batch {i+1}/5:")
            batch = next(it)
            print(f"     Shape: {batch.shape}")
            print(f"     Token range: [{batch.min().item()}, {batch.max().item()}]")

            # Verify tokens are valid
            if batch.max().item() >= vocab_size:
                print(f"     ⚠️ WARNING: Token exceeds vocab_size!")
                return False

            # Decode first sequence
            decoded = tokenizer.decode(batch[0], skip_special_tokens=False)
            print(f"     Preview: {decoded[:150]}...")

            success_count += 1

        except Exception as e:
            print(f"     ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*60)
    print(f"✅ Quick test passed! ({success_count}/5 batches successful)")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_dataset_quick()
    if not success:
        print("\n❌ Quick test failed!")
        sys.exit(1)
    else:
        print("\n✨ Dataset is working! Ready for full test or training.")
        print("\nNext steps:")
        print("  1. Run comprehensive test: python test_chimera_comprehensive.py")
        print("  2. Or start training: python scripts/train_chimera.py")
