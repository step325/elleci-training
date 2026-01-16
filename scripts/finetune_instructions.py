#!/usr/bin/env python3
"""
Elleci V1 - Instruction Fine-Tuning Script

Fine-tuning del modello base per instruction-following.
Ottimizzato per RTX 4090 (24GB VRAM) su vast.ai.

Datasets:
- Italian: Local instructions + Fauno StackOverflow/Quora (~109K)
- English: OpenHermes 2.5 (1M high-quality instruction samples)

Usage:
    # Fine-tuning standard (15K steps)
    python scripts/finetune_instructions.py --checkpoint checkpoints/elleci_v1_final.pth

    # Quick test
    python scripts/finetune_instructions.py --checkpoint checkpoints/elleci_v1_final.pth --dry-run

    # Custom steps
    python scripts/finetune_instructions.py --checkpoint checkpoints/elleci_v1_final.pth --steps 20000
"""

import os
import sys
import math
import time
import gc
import glob
import json
import random
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# Add root to path
sys.path.append(os.getcwd())

from src.config import ElleciConfig
from src.model import Elleci

# ======== CUDA OPTIMIZATIONS ========
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(False)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class FinetuneConfig:
    """Configuration for instruction fine-tuning"""
    # Training
    total_steps: int = 15000
    warmup_steps: int = 500
    save_interval: int = 2500
    eval_interval: int = 500

    # Batch (optimized for 24GB VRAM)
    batch_size: int = 2           # Per-GPU batch size
    grad_accum_steps: int = 32    # Effective batch = 64
    max_seq_len: int = 1024       # Full context

    # Learning rate (lower for fine-tuning)
    learning_rate: float = 2e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01

    # Dataset mix
    instruction_ratio: float = 0.90  # 90% instructions, 10% text for stability
    italian_ratio: float = 0.40      # 40% Italian, 60% English instructions

    # Curriculum: start with simple, add complex
    curriculum_stages: List[tuple] = None

    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = [
                (0.0, 0.3, "simple"),    # 0-30%: Simple Q&A
                (0.3, 0.7, "medium"),    # 30-70%: Medium complexity
                (0.7, 1.0, "complex"),   # 70-100%: Complex reasoning
            ]


# ============================================================
# INSTRUCTION DATASET
# ============================================================

class InstructionDataset(IterableDataset):
    """
    Dataset for instruction fine-tuning with curriculum learning.

    Sources:
    - Italian: Local JSONL + Fauno StackOverflow + Fauno Quora
    - English: OpenHermes 2.5 (teknium/OpenHermes-2.5)
    """

    def __init__(
        self,
        tokenizer,
        config: FinetuneConfig,
        current_stage: str = "simple"
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.current_stage = current_stage
        self.max_length = config.max_seq_len
        self.batch_size = config.batch_size

        # Load local Italian instructions
        self.it_instructions = self._load_local_instructions()
        print(f"[IT] Loaded {len(self.it_instructions)} local Italian instructions")

        # Streaming iterators (lazy loaded)
        self._openhermes_iter = None
        self._stackoverflow_iter = None
        self._quora_iter = None

        # Complexity filters
        self._setup_complexity_filters()

    def _setup_complexity_filters(self):
        """Setup filters based on instruction complexity"""
        # Simple: short, direct Q&A
        # Medium: explanations, lists
        # Complex: reasoning, code, multi-step

        self.complexity_keywords = {
            "simple": {
                "max_length": 500,  # Increased length, removed keywords requirement
                "keywords": [],
            },
            "medium": {
                "max_length": 1500, # Increased length
                "keywords": [],
            },
            "complex": {
                "max_length": 4000,
                "keywords": [],
            },
        }

    def _load_local_instructions(self) -> List[Dict]:
        """Load Italian instructions from local JSONL files"""
        instructions = []

        files = glob.glob("data/chimera_instructions_final.jsonl")
        if not files:
            files = glob.glob("data/elleci_instructions.jsonl")

        for fpath in files:
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                instructions.append(data)
                            except:
                                pass

        return instructions

    def _format_chatml(self, user: str, assistant: str) -> str:
        """Format as ChatML"""
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"

    def _get_openhermes_stream(self):
        """Get OpenHermes 2.5 stream (1M+ high-quality instructions)"""
        from datasets import load_dataset

        print("[EN] Loading OpenHermes 2.5 stream...")
        try:
            ds = load_dataset(
                "teknium/OpenHermes-2.5",
                split="train",
                streaming=True
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=10000))
        except Exception as e:
            print(f"[!] OpenHermes failed: {e}")
            print("    Trying SlimOrca fallback...")
            try:
                ds = load_dataset(
                    "Open-Orca/SlimOrca",
                    split="train",
                    streaming=True
                )
                return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=10000))
            except Exception as e2:
                print(f"[!] SlimOrca also failed: {e2}")
                return None

    def _get_stackoverflow_stream(self):
        """Get StackOverflow Italian stream"""
        from datasets import load_dataset

        print("[IT] Loading Fauno StackOverflow stream...")
        try:
            ds = load_dataset(
                "andreabac3/StackOverflow-Italian-Fauno-Baize",
                split="train",
                streaming=True
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=5000))
        except Exception as e:
            print(f"[!] StackOverflow Italian failed: {e}")
            return None

    def _get_quora_stream(self):
        """Get Quora Italian stream"""
        from datasets import load_dataset

        print("[IT] Loading Fauno Quora stream...")
        try:
            ds = load_dataset(
                "andreabac3/Quora-Italian-Fauno-Baize",
                split="train",
                streaming=True
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=5000))
        except Exception as e:
            print(f"[!] Quora Italian failed: {e}")
            return None

    def _filter_by_complexity(self, text: str, response: str) -> bool:
        """Check if sample matches current complexity stage"""
        stage_config = self.complexity_keywords.get(self.current_stage, {})
        max_len = stage_config.get("max_length", 2000)
        keywords = stage_config.get("keywords", [])

        # Check response length
        if len(response) > max_len:
            return False

        # For "complex" stage, accept everything
        if self.current_stage == "complex":
            return True

        # Check keywords (case insensitive)
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def _get_english_sample(self) -> Optional[str]:
        """Get English instruction sample from OpenHermes"""
        try:
            if self._openhermes_iter is None:
                self._openhermes_iter = self._get_openhermes_stream()
                if self._openhermes_iter is None:
                    return None

            item = next(self._openhermes_iter)

            # OpenHermes format: conversations list
            conversations = item.get("conversations", [])
            if len(conversations) >= 2:
                user_msg = conversations[0].get("value", "")
                assistant_msg = conversations[1].get("value", "")

                # Filter by complexity
                if not self._filter_by_complexity(user_msg, assistant_msg):
                    return None

                return self._format_chatml(user_msg, assistant_msg)

            return None

        except StopIteration:
            self._openhermes_iter = self._get_openhermes_stream()
            return None
        except Exception as e:
            return None

    def _get_italian_sample(self) -> Optional[str]:
        """Get Italian instruction sample"""
        source_choice = random.random()

        try:
            if source_choice < 0.4:
                # Local instructions (40%)
                if not self.it_instructions:
                    return None

                item = random.choice(self.it_instructions)
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")

                full_inst = f"{inst}\n{inp}".strip() if inp else inst

                if not self._filter_by_complexity(full_inst, out):
                    return None

                return self._format_chatml(full_inst, out)

            elif source_choice < 0.7:
                # StackOverflow Italian (30%)
                if self._stackoverflow_iter is None:
                    self._stackoverflow_iter = self._get_stackoverflow_stream()

                if self._stackoverflow_iter is None:
                    return self._get_italian_sample()  # Fallback

                item = next(self._stackoverflow_iter)
                text = item.get("input", "")

                if text and len(text) > 50:
                    return text
                return None

            else:
                # Quora Italian (30%)
                if self._quora_iter is None:
                    self._quora_iter = self._get_quora_stream()

                if self._quora_iter is None:
                    return self._get_italian_sample()  # Fallback

                item = next(self._quora_iter)
                text = item.get("input", "")

                if text and len(text) > 50:
                    return text
                return None

        except StopIteration:
            self._stackoverflow_iter = None
            self._quora_iter = None
            return None
        except Exception as e:
            return None

    def set_stage(self, stage: str):
        """Update curriculum stage"""
        if stage in self.complexity_keywords:
            self.current_stage = stage
            print(f"[Curriculum] Stage changed to: {stage}")

    def __iter__(self):
        """Yield batched tokenized sequences"""
        batch = []
        token_buffer = []

        while True:
            # Select instruction vs text
            if random.random() < self.config.instruction_ratio:
                # Instruction sample
                if random.random() < self.config.italian_ratio:
                    text = self._get_italian_sample()
                else:
                    text = self._get_english_sample()
            else:
                # Stability: occasional raw text (not implemented, skip)
                text = None

            if not text:
                continue

            try:
                tokens = self.tokenizer.encode(text)

                if len(tokens) < 20:
                    continue

                # Add EOS
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    tokens.append(eos_id)

                # Packing
                token_buffer.extend(tokens)

                while len(token_buffer) >= self.max_length:
                    sample_tokens = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]

                    batch.append(torch.tensor(sample_tokens, dtype=torch.long))

                    if len(batch) >= self.batch_size:
                        yield self._collate(batch)
                        batch = []

            except Exception as e:
                continue

    def _collate(self, batch):
        """Collate with padding"""
        max_len = max(len(x) for x in batch)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)

        for i, x in enumerate(batch):
            padded[i, :len(x)] = x

        return padded


# ============================================================
# EVALUATION
# ============================================================

class InstructionEvaluator:
    """Evaluate instruction-following capability"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Test prompts for evaluation
        self.eval_prompts = [
            # Greetings / Basic
            ("<|im_start|>user\nCiao, come stai?<|im_end|>\n<|im_start|>assistant\n", ["bene", "grazie", "aiutarti"], "chat"),
            ("<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n", ["good", "thanks", "help"], "chat"),
            
            # Factual
            ("<|im_start|>user\nQual è la capitale della Francia?<|im_end|>\n<|im_start|>assistant\n", ["parigi", "paris"], "fact"),
            ("<|im_start|>user\nWho wrote 'Romeo and Juliet'?<|im_end|>\n<|im_start|>assistant\n", ["shakespeare"], "fact"),
            ("<|im_start|>user\nWhat is the boiling point of water?<|im_end|>\n<|im_start|>assistant\n", ["100", "degrees", "celsius"], "fact"),
            
            # Reasoning
            ("<|im_start|>user\nSe ho 3 mele e ne mangio una, quante me ne restano?<|im_end|>\n<|im_start|>assistant\n", ["2", "due"], "logic"),
            ("<|im_start|>user\nIf I have 5 apples and buy 2 more, how many do I have?<|im_end|>\n<|im_start|>assistant\n", ["7", "seven"], "logic"),
            
            # Translation
            ("<|im_start|>user\nTranslate 'Dog' into Italian.<|im_end|>\n<|im_start|>assistant\n", ["cane"], "trans"),
            ("<|im_start|>user\nTraduci 'Gatto' in inglese.<|im_end|>\n<|im_start|>assistant\n", ["cat"], "trans"),
            
            # Coding (Simple)
            ("<|im_start|>user\nWrite a Python function to add two numbers.<|im_end|>\n<|im_start|>assistant\n", ["def", "return", "+", "add"], "code"),
            
            # Instruction Following
            ("<|im_start|>user\nList three primary colors.<|im_end|>\n<|im_start|>assistant\n", ["red", "blue", "yellow", "rosso", "blu", "giallo"], "instruct"),
            ("<|im_start|>user\nElenca i tre colori primari.<|im_end|>\n<|im_start|>assistant\n", ["rosso", "blu", "giallo"], "instruct"),
        ]

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metrics"""
        self.model.eval()

        results = {"total": 0, "correct": 0}
        category_results = {}

        for prompt, expected_keywords, category in self.eval_prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            try:
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.3,
                    top_k=20
                )

                response = self.tokenizer.decode(
                    output[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                ).lower()

                # Check keywords
                found = any(kw.lower() in response for kw in expected_keywords)

                results["total"] += 1
                if found:
                    results["correct"] += 1

                # Category tracking
                if category not in category_results:
                    category_results[category] = {"total": 0, "correct": 0}
                category_results[category]["total"] += 1
                if found:
                    category_results[category]["correct"] += 1

            except Exception as e:
                results["total"] += 1

        self.model.train()

        # Calculate accuracy
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0

        metrics = {
            "eval/accuracy": accuracy,
            "eval/correct": results["correct"],
            "eval/total": results["total"],
        }

        # Add per-category metrics
        for cat, data in category_results.items():
            cat_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
            metrics[f"eval/{cat}_acc"] = cat_acc

        return metrics


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def get_cosine_schedule(step, total_steps, warmup_steps, min_lr_ratio=0.1):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    # Don't go below min_lr_ratio of peak LR
    return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay


def get_curriculum_stage(step, total_steps, stages):
    """Get current curriculum stage based on training progress"""
    progress = step / total_steps

    for start, end, stage_name in stages:
        if start <= progress < end:
            return stage_name

    return stages[-1][2]  # Default to last stage


def print_vram_status():
    """Print current VRAM usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


def cleanup_old_checkpoints(checkpoint_dir="checkpoints", prefix="elleci_instruct_", keep_last=3):
    """Keep only the last N checkpoints"""
    if not os.path.exists(checkpoint_dir):
        return

    checkpoints = glob.glob(os.path.join(checkpoint_dir, f"{prefix}*.pth"))

    if len(checkpoints) <= keep_last:
        return

    checkpoints.sort(key=os.path.getmtime)

    for ckpt in checkpoints[:-keep_last]:
        try:
            os.remove(ckpt)
            print(f"[Cleanup] Deleted: {os.path.basename(ckpt)}")
        except Exception as e:
            pass


# ============================================================
# MAIN TRAINING
# ============================================================

def train(args):
    """Main fine-tuning loop"""

    config = FinetuneConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
    )

    if args.dry_run:
        print("[DRY RUN MODE]")
        config.total_steps = 50
        config.save_interval = 10
        config.eval_interval = 10
        config.warmup_steps = 5

    print("=" * 60)
    print("  ELLECI INSTRUCTION FINE-TUNING")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Total steps: {config.total_steps}")
    print(f"  Batch size: {config.batch_size} x {config.grad_accum_steps} = {config.batch_size * config.grad_accum_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Instruction ratio: {config.instruction_ratio:.0%}")
    print("=" * 60)

    # 1. Load tokenizer
    tokenizer_path = "tokenizer_elleci_v1/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "tokenizer_chimera_v2_patched/tokenizer.json"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"

    vocab_size = ((len(tokenizer.get_vocab()) + 127) // 128) * 128
    print(f"[Tokenizer] Vocab size: {vocab_size}")

    # 2. Load model
    model_config = ElleciConfig(
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=2048,
        n_layers=24,
        use_router=False,
    )
    model_config.mamba.use_mamba2 = True
    model_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Elleci(model_config)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"[Model] Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        del state_dict
    else:
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = model.to(model_config.device)

    # Enable memory optimizations
    model.gradient_checkpointing_enable()
    print("[Memory] Gradient checkpointing enabled")

    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[Model] Parameters: {params:.2f}B")
    print_vram_status()

    # 3. Optimizer
    if BNB_AVAILABLE:
        print("[Optimizer] Using 8-bit AdamW (bitsandbytes)")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        print("[Optimizer] Using standard AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_cosine_schedule(
            step, config.total_steps, config.warmup_steps,
            min_lr_ratio=config.min_lr / config.learning_rate
        )
    )

    # 4. Mixed precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[Precision] Using {'bfloat16' if use_bf16 else 'float16'}")

    scaler = torch.amp.GradScaler('cuda', enabled=(not use_bf16))

    # 5. Dataset
    current_stage = get_curriculum_stage(0, config.total_steps, config.curriculum_stages)
    dataset = InstructionDataset(tokenizer, config, current_stage)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)
    data_iter = iter(dataloader)

    # 6. Evaluator
    evaluator = InstructionEvaluator(model, tokenizer, model_config.device)

    # 7. WandB
    if WANDB_AVAILABLE and not args.no_wandb and not args.dry_run:
        wandb.init(
            project="Elleci-Finetune",
            name=f"instruct-{config.total_steps}steps",
            config={
                "total_steps": config.total_steps,
                "batch_size": config.batch_size,
                "grad_accum": config.grad_accum_steps,
                "learning_rate": config.learning_rate,
            }
        )

    # 8. Training loop
    os.makedirs("checkpoints", exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"\n[Training] Starting from step 0...")
    pbar = tqdm(range(config.total_steps), desc="Fine-tuning")

    accum_loss = 0
    best_accuracy = 0

    for step in pbar:
        # Curriculum update
        new_stage = get_curriculum_stage(step, config.total_steps, config.curriculum_stages)
        if new_stage != current_stage:
            current_stage = new_stage
            dataset.set_stage(current_stage)

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(model_config.device)
        targets = batch.clone()

        # Forward
        with torch.amp.autocast('cuda', dtype=dtype):
            logits, loss = model(batch, targets=targets)

        # Backward
        if use_bf16:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        accum_loss += loss.item() / config.grad_accum_steps

        # Optimizer step
        if (step + 1) % config.grad_accum_steps == 0:
            # Gradient clipping
            if use_bf16:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Log
            curr_lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Loss: {accum_loss:.4f} | LR: {curr_lr:.2e} | Stage: {current_stage}")

            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    "train/loss": accum_loss,
                    "train/lr": curr_lr,
                    "train/stage": current_stage,
                }, step=step)

            accum_loss = 0

        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            metrics = evaluator.evaluate()
            accuracy = metrics["eval/accuracy"]

            print(f"\n[Eval @ {step+1}] Accuracy: {accuracy:.1%} ({metrics['eval/correct']}/{metrics['eval/total']})")

            if WANDB_AVAILABLE and wandb.run:
                wandb.log(metrics, step=step)

            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "checkpoints/elleci_instruct_best.pth")
                print(f"[Save] New best model! Accuracy: {accuracy:.1%}")

            model.train()

        # Save checkpoint
        if (step + 1) % config.save_interval == 0:
            checkpoint_path = f"checkpoints/elleci_instruct_step_{step+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\n[Save] Checkpoint: {checkpoint_path}")
            cleanup_old_checkpoints()

        # Memory cleanup
        if (step + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Final save
    torch.save(model.state_dict(), "checkpoints/elleci_instruct_final.pth")
    print("\n[Training Complete]")
    print(f"  Final checkpoint: checkpoints/elleci_instruct_final.pth")
    print(f"  Best accuracy: {best_accuracy:.1%}")

    # Final evaluation
    print("\n[Final Evaluation]")
    metrics = evaluator.evaluate()
    print(f"  Accuracy: {metrics['eval/accuracy']:.1%}")

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Elleci for instruction-following")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--steps", type=int, default=15000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
