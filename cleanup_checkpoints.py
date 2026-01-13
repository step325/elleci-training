"""
Cleanup Old Checkpoints - Standalone Script
Keeps only the most recent N checkpoints to free disk space.

Usage:
    python cleanup_checkpoints.py --dir checkpoints --keep 3
    python cleanup_checkpoints.py --dir /workspace/elleci-training/checkpoints --keep 3
"""
import os
import glob
import argparse

def cleanup_checkpoints(checkpoint_dir, keep_last=3, dry_run=False):
    """
    Keep only the last N checkpoints to save disk space.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of most recent checkpoints to keep
        dry_run: If True, only show what would be deleted
    """
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Directory not found: {checkpoint_dir}")
        return

    # Find all checkpoint files (support both patterns)
    patterns = [
        os.path.join(checkpoint_dir, "elleci_step_*.pth"),
        os.path.join(checkpoint_dir, "chimera_step_*.pth"),
        os.path.join(checkpoint_dir, "*_step_*.pth"),
    ]

    all_checkpoints = []
    for pattern in patterns:
        all_checkpoints.extend(glob.glob(pattern))

    # Remove duplicates
    all_checkpoints = list(set(all_checkpoints))

    if not all_checkpoints:
        print(f"ℹ️  No checkpoints found in {checkpoint_dir}")
        return

    print(f"📊 Found {len(all_checkpoints)} checkpoint(s)")

    if len(all_checkpoints) <= keep_last:
        print(f"✅ Already at or below limit ({keep_last} checkpoints)")
        return

    # Sort by modification time (oldest first)
    all_checkpoints.sort(key=os.path.getmtime)

    # Calculate sizes
    to_delete = all_checkpoints[:-keep_last]
    to_keep = all_checkpoints[-keep_last:]

    total_size = 0
    for ckpt in to_delete:
        try:
            size = os.path.getsize(ckpt)
            total_size += size
        except:
            pass

    total_size_gb = total_size / (1024**3)

    print(f"\n🗑️  Will delete {len(to_delete)} old checkpoint(s)")
    print(f"💾 Will free ~{total_size_gb:.2f} GB")
    print(f"✅ Will keep {len(to_keep)} most recent checkpoint(s)")

    print("\n📦 Keeping:")
    for ckpt in to_keep:
        print(f"   ✓ {os.path.basename(ckpt)}")

    print("\n🗑️  Deleting:")
    for ckpt in to_delete:
        size_mb = os.path.getsize(ckpt) / (1024**2)
        print(f"   ✗ {os.path.basename(ckpt)} ({size_mb:.1f} MB)")

    if dry_run:
        print("\n⚠️  DRY RUN - No files were deleted")
        return

    # Delete old checkpoints
    deleted_count = 0
    for ckpt in to_delete:
        try:
            os.remove(ckpt)
            deleted_count += 1
        except Exception as e:
            print(f"   ❌ Failed to delete {os.path.basename(ckpt)}: {e}")

    print(f"\n✅ Deleted {deleted_count}/{len(to_delete)} checkpoint(s)")
    print(f"💾 Freed ~{total_size_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup old training checkpoints")
    parser.add_argument("--dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--keep", type=int, default=3, help="Number of recent checkpoints to keep")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")

    args = parser.parse_args()

    print("🧹 Checkpoint Cleanup Tool")
    print("=" * 60)
    print(f"Directory: {args.dir}")
    print(f"Keep last: {args.keep}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60 + "\n")

    cleanup_checkpoints(args.dir, args.keep, args.dry_run)
