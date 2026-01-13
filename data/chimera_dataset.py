"""
Chimera Dataset - Direct Streaming (No Buffer)
Mixes English (Cosmopedia) + Italian (Wikipedia + Instructions) with phase-based ratios.
"""
import torch
from torch.utils.data import IterableDataset
import random
from datasets import load_dataset
import json
import os
import glob


class ChimeraDataset(IterableDataset):
    """
    Streaming dataset for Chimera V1 training.

    Phase 1 (Knowledge Acquisition - 90% of training):
        - 55% English Cosmopedia V2
        - 35% Italian Wikipedia
        - 10% Italian Instructions

    Phase 2 (Instruction Alignment - 10% of training):
        - 20% English Cosmopedia V2
        - 25% Italian Wikipedia
        - 55% Italian Instructions
    """

    def __init__(self, tokenizer, phase=1, max_length=512, batch_size=32):
        self.tokenizer = tokenizer
        self.phase = phase
        self.max_length = max_length
        self.batch_size = batch_size

        # Define mixing ratios based on phase
        if phase == 1:
            self.ratios = {
                'en_cosmo': 0.55,
                'it_wiki': 0.35,
                'it_instruct': 0.10
            }
        elif phase == 2:
            self.ratios = {
                'en_cosmo': 0.20,
                'it_wiki': 0.25,
                'it_instruct': 0.55
            }
        else:
            raise ValueError(f"Unknown phase: {phase}")

        print(f"🦁 ChimeraDataset Phase {phase} | Max Length: {max_length}")
        print(f"📊 Ratios: EN={self.ratios['en_cosmo']:.0%}, IT_Wiki={self.ratios['it_wiki']:.0%}, IT_Instr={self.ratios['it_instruct']:.0%}")

        # Load local instructions (fast, in-memory)
        self.it_instructions = self._load_local_instructions()
        print(f"🇮🇹 Loaded {len(self.it_instructions)} Italian instructions")

        # Streaming iterators (created lazily in __iter__)
        self._en_iter = None
        self._it_iter = None

    def _load_local_instructions(self):
        """Load Italian instructions from local JSONL files."""
        instructions = []

        # Try both instruction files
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

        if not instructions:
            raise FileNotFoundError("No Italian instruction files found! Need chimera_instructions_final.jsonl or elleci_instructions.jsonl")

        return instructions

    def _format_chatml(self, user, assistant):
        """Format instruction as ChatML."""
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"

    def _get_en_stream(self):
        """Get English Cosmopedia V2 stream."""
        print("📥 Loading Cosmopedia V2 stream...")
        try:
            ds = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "cosmopedia-v2",
                split="train",
                streaming=True
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"⚠️ Error loading Cosmopedia: {e}")
            raise

    def _get_it_stream(self):
        """Get Italian Wikipedia stream."""
        print("📥 Loading Italian Wikipedia stream...")
        try:
            # Try primary Wikipedia source
            ds = load_dataset(
                "wikimedia/wikipedia",
                "20231101.it",
                split="train",
                streaming=True
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"⚠️ Primary Wikipedia failed: {e}")
            try:
                # Fallback to alternative source
                ds = load_dataset(
                    "graelo/wikipedia",
                    "20230601.it",
                    split="train",
                    streaming=True
                )
                return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
            except Exception as e2:
                print(f"⚠️ Fallback Wikipedia also failed: {e2}")
                raise

    def _get_next_sample(self, source):
        """
        Get next sample from specified source.
        Returns text string or None if failed.
        """
        try:
            if source == 'en_cosmo':
                # English Cosmopedia
                if self._en_iter is None:
                    self._en_iter = self._get_en_stream()

                item = next(self._en_iter)
                text = item.get("text", "")
                if len(text) < 50:  # Skip very short texts
                    return None
                return text

            elif source == 'it_wiki':
                # Italian Wikipedia
                if self._it_iter is None:
                    self._it_iter = self._get_it_stream()

                item = next(self._it_iter)
                text = item.get("text", "")
                if len(text) < 50:
                    return None
                return text

            elif source == 'it_instruct':
                # Italian Instructions (local)
                item = random.choice(self.it_instructions)
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")

                full_inst = f"{inst}\n{inp}".strip()
                text = self._format_chatml(full_inst, out)
                return text

        except StopIteration:
            # Stream exhausted, reset it
            if source == 'en_cosmo':
                self._en_iter = self._get_en_stream()
            elif source == 'it_wiki':
                self._it_iter = self._get_it_stream()
            return None
        except Exception as e:
            print(f"⚠️ Error getting sample from {source}: {e}")
            return None

    def __iter__(self):
        """
        Yield batches of tokenized sequences.
        Direct streaming, no buffering.
        """
        batch = []

        while True:
            # Select source based on ratios
            r = random.random()

            if r < self.ratios['en_cosmo']:
                source = 'en_cosmo'
            elif r < self.ratios['en_cosmo'] + self.ratios['it_wiki']:
                source = 'it_wiki'
            else:
                source = 'it_instruct'

            # Get text from selected source
            text = self._get_next_sample(source)
            if not text:
                continue

            # Tokenize
            try:
                tokens = self.tokenizer.encode(text)

                # Add EOS token
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    tokens.append(eos_id)

                # Truncate if too long
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]

                # Skip very short sequences
                if len(tokens) < 10:
                    continue

                batch.append(torch.tensor(tokens, dtype=torch.long))

                # Yield batch when full
                if len(batch) >= self.batch_size:
                    yield self._collate(batch)
                    batch = []

            except Exception as e:
                # Skip problematic samples
                continue

    def _collate(self, batch):
        """
        Collate batch with padding.
        Returns: [batch_size, max_len] tensor
        """
        max_len = max(len(x) for x in batch)

        # Get pad token
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        # Create padded tensor
        padded_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)

        for i, x in enumerate(batch):
            padded_batch[i, :len(x)] = x

        return padded_batch
