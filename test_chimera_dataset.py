"""
Quick test for ChimeraDataset streaming.
"""
import sys
import os
from transformers import PreTrainedTokenizerFast

sys.path.append(os.getcwd())

from data.chimera_dataset import ChimeraDataset

def test_dataset():
    print("🔍 Loading Tokenizer...")
    tokenizer_path = "tokenizer_chimera_v2_patched/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        return

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"

    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer.get_vocab())})")

    print("\n🔍 Creating ChimeraDataset (Phase 1)...")
    dataset = ChimeraDataset(
        tokenizer,
        phase=1,
        max_length=256,
        batch_size=4
    )

    print("\n🔍 Testing streaming (fetching 3 batches)...")
    it = iter(dataset)

    for i in range(3):
        try:
            print(f"\n  Batch {i+1}:")
            batch = next(it)
            print(f"    Shape: {batch.shape}")
            print(f"    Min token: {batch.min().item()}, Max token: {batch.max().item()}")
            print(f"    Sample (first 20 tokens): {batch[0, :20].tolist()}")

            # Decode first sequence
            decoded = tokenizer.decode(batch[0], skip_special_tokens=False)
            print(f"    Decoded preview: {decoded[:200]}...")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n✅ Dataset test completed!")

if __name__ == "__main__":
    test_dataset()
