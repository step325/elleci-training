"""
Debug script for ChimeraDataset.
Isolates the data loading logic to identify where it hangs.
"""
import sys
import os
import time
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader

# Add root to path
sys.path.append(os.getcwd())

from data.chimera_dataset import ChimeraDataset

def test_dataset():
    print("🔍 initializing Tokenizer...")
    tokenizer_path = "tokenizer_chimera_v2_patched/tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    
    print("🔍 initializing ChimeraDataset (Phase 1)...")
    dataset = ChimeraDataset(tokenizer, phase=1, max_length=512, batch_size=4)
    
    # Test 1: Only Instruct (Local)
    print("\n[TEST 1] Only Instruct (Local)")
    dataset.ratios = {'en_cosmo': 0.0, 'it_wiki': 0.0, 'it_instruct': 1.0}
    it = iter(dataset)
    try:
        print("  next(it)...")
        b = next(it)
        print("  ✅ Access OK")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Test 2: Only Wiki (IT)
    print("\n[TEST 2] Only Wiki (IT)")
    dataset.ratios = {'en_cosmo': 0.0, 'it_wiki': 1.0, 'it_instruct': 0.0}
    it = iter(dataset)
    try:
        print("  next(it)...")
        b = next(it)
        print("  ✅ Access OK")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Test 3: Only Cosmopedia (EN)
    print("\n[TEST 3] Only Cosmopedia (EN)")
    dataset.ratios = {'en_cosmo': 1.0, 'it_wiki': 0.0, 'it_instruct': 0.0}
    it = iter(dataset)
    try:
        print("  next(it)...")
        b = next(it)
        print("  ✅ Access OK")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        
    return

if __name__ == "__main__":
    test_dataset()
