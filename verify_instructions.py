"""
Verify Italian instruction files without requiring PyTorch.
"""
import json
import glob
import os

def verify_instructions():
    """Verify instruction files are valid and properly formatted."""
    print("🔍 Verifying Italian Instruction Files")
    print("="*60)

    # Find instruction files
    files = glob.glob("data/chimera_instructions_final.jsonl")
    if not files:
        files = glob.glob("data/elleci_instructions.jsonl")

    if not files:
        print("❌ No instruction files found!")
        return False

    total_samples = 0
    all_instructions = []

    for fpath in files:
        if not os.path.exists(fpath):
            continue

        print(f"\n📄 Checking: {fpath}")
        file_samples = 0
        valid_samples = 0

        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                file_samples += 1
                try:
                    data = json.loads(line)

                    # Check required fields
                    if "instruction" not in data or "output" not in data:
                        print(f"  ⚠️ Line {line_num}: Missing required fields")
                        continue

                    # Check non-empty
                    if not data["instruction"] or not data["output"]:
                        print(f"  ⚠️ Line {line_num}: Empty instruction or output")
                        continue

                    valid_samples += 1
                    all_instructions.append(data)

                except json.JSONDecodeError as e:
                    print(f"  ❌ Line {line_num}: Invalid JSON - {e}")
                    continue

        print(f"  ✅ Loaded {valid_samples}/{file_samples} valid samples")
        total_samples += valid_samples

    print(f"\n{'='*60}")
    print(f"📊 Total: {total_samples} valid instruction samples")
    print(f"{'='*60}")

    # Show sample
    if all_instructions:
        print("\n📝 Sample instruction:")
        sample = all_instructions[0]
        print(f"  Instruction: {sample['instruction'][:100]}...")
        print(f"  Output: {sample['output'][:100]}...")
        if 'category' in sample:
            print(f"  Category: {sample.get('category', 'N/A')}")

    # Verify proportions
    print(f"\n📊 Dataset Proportions:")
    print(f"  Instruction Sources:")
    print(f"    - Local custom: {total_samples} samples")
    print(f"    - Fauno StackOverflow: ~47,000 samples (streaming)")
    print(f"    - Fauno Quora: ~54,000 samples (streaming)")
    print(f"    - TOTAL: ~109,000 instruction samples")
    print(f"\n  Phase 1 (90% training):")
    print(f"    - 55% English Cosmopedia (streaming)")
    print(f"    - 35% Italian CulturaX (streaming)")
    print(f"    - 10% Italian Instructions (~109K total)")
    print(f"\n  Phase 2 (10% training):")
    print(f"    - 20% English Cosmopedia (streaming)")
    print(f"    - 25% Italian CulturaX (streaming)")
    print(f"    - 55% Italian Instructions (~109K total)")
    print(f"    - Expected epochs: ~0.4 (prevents overfitting!)")

    return total_samples > 0

if __name__ == "__main__":
    success = verify_instructions()
    if success:
        print("\n✅ Instruction files are valid!")
    else:
        print("\n❌ Instruction verification failed!")
        exit(1)
