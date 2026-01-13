"""
Chimera Dataset - Direct Streaming (No Buffer)
Mixes English (Cosmopedia) + Italian (CulturaX + Instructions) with phase-based ratios.

Data Sources:
- English: HuggingFaceTB/smollm-corpus (Cosmopedia V2)
- Italian: uonlp/CulturaX (200GB+ cleaned, 41B words - highest quality Italian corpus)
- Instructions:
  - Local JSONL files (7,673 Italian instruction-response pairs)
  - Fauno StackOverflow Italian (~47K technical/coding samples)
  - Fauno Quora Italian (~54K conversational samples)
  - Total: ~109K instruction samples
"""
import torch
from torch.utils.data import IterableDataset
import random
from datasets import load_dataset
import json
import os
import glob


class EllediDataset(IterableDataset):
    """
    Streaming dataset for Chimera V1 training.

    Phase 1 (Knowledge Acquisition - 90% of training):
        - 55% English Cosmopedia V2 (educational content)
        - 35% Italian CulturaX (200GB+ cleaned Italian corpus)
        - 10% Italian Instructions (~109K samples: custom + Fauno StackOverflow + Fauno Quora)

    Phase 2 (Instruction Alignment - 10% of training):
        - 20% English Cosmopedia V2
        - 25% Italian CulturaX
        - 55% Italian Instructions (~109K samples: custom + Fauno StackOverflow + Fauno Quora)
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
        print(f"📊 Ratios: EN_Cosmo={self.ratios['en_cosmo']:.0%}, IT_CulturaX={self.ratios['it_wiki']:.0%}, IT_Instr={self.ratios['it_instruct']:.0%}")

        # Load local instructions (fast, in-memory)
        self.it_instructions = self._load_local_instructions()
        print(f"🇮🇹 Loaded {len(self.it_instructions)} local Italian instructions")
        print(f"📚 Instruction sources: Local (7.7K) + Fauno StackOverflow (47K) + Fauno Quora (54K) = ~109K total")

        # Streaming iterators (created lazily in __iter__)
        self._en_iter = None
        self._it_iter = None
        self._stackoverflow_iter = None
        self._quora_iter = None

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
        """Get Italian text stream from CulturaX (highest quality Italian corpus)."""
        print("📥 Loading Italian CulturaX stream (200GB+ cleaned Italian)...")
        try:
            # CulturaX - Best Italian dataset (41B words, deeply cleaned)
            ds = load_dataset(
                "uonlp/CulturaX",
                "it",
                split="train",
                streaming=True
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"⚠️ CulturaX failed, trying Wikipedia fallback: {e}")
            try:
                # Fallback 1: Wikipedia IT
                ds = load_dataset(
                    "wikimedia/wikipedia",
                    "20231101.it",
                    split="train",
                    streaming=True
                )
                return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
            except Exception as e2:
                print(f"⚠️ Wikipedia also failed, trying secondary fallback: {e2}")
                try:
                    # Fallback 2: Alternative Wikipedia
                    ds = load_dataset(
                        "graelo/wikipedia",
                        "20230601.it",
                        split="train",
                        streaming=True
                    )
                    return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
                except Exception as e3:
                    print(f"⚠️ All Italian sources failed: {e3}")
                    raise

    def _get_stackoverflow_stream(self):
        """Get StackOverflow Italian instruction stream from Fauno dataset."""
        print("📥 Loading Fauno StackOverflow Italian stream (~47K samples)...")
        try:
            ds = load_dataset(
                "andreabac3/StackOverflow-Italian-Fauno-Baize",
                split="train",
                streaming=True
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"⚠️ StackOverflow Italian dataset failed to load: {e}")
            print("   Continuing with local instructions only...")
            return None

    def _get_quora_stream(self):
        """Get Quora Italian instruction stream from Fauno dataset."""
        print("📥 Loading Fauno Quora Italian stream (~54K samples)...")
        try:
            ds = load_dataset(
                "andreabac3/Quora-Italian-Fauno-Baize",
                split="train",
                streaming=True
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"⚠️ Quora Italian dataset failed to load: {e}")
            print("   Continuing with local instructions only...")
            return None

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
                # Italian Instructions - randomly pick from 3 sources
                source_choice = random.random()

                if source_choice < 0.33:
                    # Local custom instructions
                    item = random.choice(self.it_instructions)
                    inst = item.get("instruction", "")
                    inp = item.get("input", "")
                    out = item.get("output", "")
                    full_inst = f"{inst}\n{inp}".strip()
                    text = self._format_chatml(full_inst, out)
                    return text

                elif source_choice < 0.66:
                    # StackOverflow Italian (Fauno)
                    if self._stackoverflow_iter is None:
                        self._stackoverflow_iter = self._get_stackoverflow_stream()

                    if self._stackoverflow_iter is None:
                        # Fallback to local if stream failed
                        return self._get_next_sample('it_instruct')

                    item = next(self._stackoverflow_iter)
                    # Fauno datasets have 'input' field with full conversation
                    text = item.get("input", "")
                    if not text or len(text) < 20:
                        return None
                    return text

                else:
                    # Quora Italian (Fauno)
                    if self._quora_iter is None:
                        self._quora_iter = self._get_quora_stream()

                    if self._quora_iter is None:
                        # Fallback to local if stream failed
                        return self._get_next_sample('it_instruct')

                    item = next(self._quora_iter)
                    # Fauno datasets have 'input' field with full conversation
                    text = item.get("input", "")
                    if not text or len(text) < 20:
                        return None
                    return text

        except StopIteration:
            # Stream exhausted, reset it
            if source == 'en_cosmo':
                self._en_iter = self._get_en_stream()
            elif source == 'it_wiki':
                self._it_iter = self._get_it_stream()
            # Fauno streams will auto-reset on next call due to None check
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
