"""
Elleci V1 Dataset Pipeline - OPTIMIZED
Provides a streaming IterableDataset that mixes English and Italian data sources.
Uses local buffering to avoid network latency bottleneck.
"""
import torch
from torch.utils.data import IterableDataset
import random
from datasets import load_dataset
import json
import os
import glob
from threading import Thread
from queue import Queue
import time

# ============ OPTIMIZATION CONSTANTS ============
BUFFER_SIZE = 500   # Reduced from 2000 to save memory
REFILL_THRESHOLD = 100  # Refill when buffer drops below this
BACKGROUND_REFILL = False  # DISABLED - was causing memory leak
# ================================================

class EllediDataset(IterableDataset):
    def __init__(self, tokenizer, phase=1, max_length=512, batch_size=32):
        self.tokenizer = tokenizer
        self.phase = phase
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Define Mixing Ratios based on Phase
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
            
        print(f"🦁 EllediDataset initialized (Phase {phase})")
        print(f"📊 Ratios: {self.ratios}")
        
        # Pre-load buffers (CRITICAL for performance)
        self._init_buffers()
        
    def _init_buffers(self):
        """Initialize local buffers for each data source."""
        print("📦 Pre-filling data buffers (this may take a minute)...")
        
        self.buffer_en = []
        self.buffer_it_wiki = []
        self.buffer_it_instruct = []
        
        # Stream iterators (will be created lazily)
        self._iter_en = None
        self._iter_it_wiki = None
        self._refilling = False
        
        # Load local instructions immediately (already fast)
        self.buffer_it_instruct = self._get_it_instruct_list()
        
        # Pre-fill remote buffers
        self._fill_buffer_en(BUFFER_SIZE)
        self._fill_buffer_it_wiki(BUFFER_SIZE)
        
        print(f"✅ Buffers ready: EN={len(self.buffer_en)}, IT_Wiki={len(self.buffer_it_wiki)}, IT_Instr={len(self.buffer_it_instruct)}")
        
    def _get_en_stream(self):
        """English Stream: Cosmopedia V2"""
        try:
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
            return iter(ds)
        except Exception as e:
            print(f"⚠️ Error loading Cosmopedia: {e}")
            return iter([])

    def _get_it_wiki_stream(self):
        """Italian Knowledge: Wikipedia"""
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.it", split="train", streaming=True)
            return iter(ds)
        except Exception as e:
            print(f"⚠️ Error loading IT Wikipedia (wikimedia): {e}")
            try:
                ds = load_dataset("graelo/wikipedia", "20230601.it", split="train", streaming=True)
                return iter(ds)
            except Exception as e2:
                print(f"⚠️ Fallback also failed: {e2}")
                return iter([])

    def _fill_buffer_en(self, count):
        """Fill English buffer with samples."""
        if self._iter_en is None:
            self._iter_en = self._get_en_stream()
        
        filled = 0
        try:
            for _ in range(count):
                item = next(self._iter_en)
                text = item.get("text", "")
                if text and len(text) > 50:
                    self.buffer_en.append(text)
                    filled += 1
        except StopIteration:
            self._iter_en = self._get_en_stream()  # Reset stream
        except Exception as e:
            print(f"⚠️ EN buffer fill error: {e}")
        
        if filled > 0:
            print(f"   📥 EN buffer +{filled} (total: {len(self.buffer_en)})")
            
    def _fill_buffer_it_wiki(self, count):
        """Fill Italian Wikipedia buffer with samples."""
        if self._iter_it_wiki is None:
            self._iter_it_wiki = self._get_it_wiki_stream()
        
        filled = 0
        try:
            for _ in range(count):
                item = next(self._iter_it_wiki)
                text = item.get("text", "")
                if text and len(text) > 50:
                    self.buffer_it_wiki.append(text)
                    filled += 1
        except StopIteration:
            self._iter_it_wiki = self._get_it_wiki_stream()  # Reset stream
        except Exception as e:
            print(f"⚠️ IT Wiki buffer fill error: {e}")
            
        if filled > 0:
            print(f"   📥 IT Wiki buffer +{filled} (total: {len(self.buffer_it_wiki)})")

    def _background_refill(self):
        """Background thread to refill buffers."""
        if self._refilling:
            return
        self._refilling = True
        
        def refill_task():
            try:
                if len(self.buffer_en) < REFILL_THRESHOLD:
                    self._fill_buffer_en(BUFFER_SIZE - len(self.buffer_en))
                if len(self.buffer_it_wiki) < REFILL_THRESHOLD:
                    self._fill_buffer_it_wiki(BUFFER_SIZE - len(self.buffer_it_wiki))
            finally:
                self._refilling = False
                
        if BACKGROUND_REFILL:
            Thread(target=refill_task, daemon=True).start()
        else:
            refill_task()

    def _get_it_instruct_list(self):
        """Italian Instructions: Local JSONL files"""
        instructions = []
        files = glob.glob("data/elleci_instructions.jsonl")
        
        if not files:
            files = glob.glob("data/chimera_instructions_final.jsonl")
            
        for fpath in files:
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                instructions.append(json.loads(line))
                            except:
                                pass
                                
        if not instructions:
            print("⚠️ No local instruction files found! Using dummy data.")
            instructions = [{"instruction": "Ciao", "output": "Ciao! Come posso aiutarti?"}]
            
        print(f"🇮🇹 Loaded {len(instructions)} local instructions")
        return instructions

    def _format_chatml(self, user, assistant):
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"

    def __iter__(self):
        batch = []
        samples_since_refill_check = 0
        
        while True:
            # Check if we need to refill buffers (every 100 samples)
            samples_since_refill_check += 1
            if samples_since_refill_check >= 100:
                samples_since_refill_check = 0
                if len(self.buffer_en) < REFILL_THRESHOLD or len(self.buffer_it_wiki) < REFILL_THRESHOLD:
                    self._background_refill()
            
            # Select source based on ratios
            r = random.random()
            text = ""
            
            threshold_en = self.ratios['en_cosmo']
            threshold_wiki = threshold_en + self.ratios['it_wiki']
            
            try:
                if r < threshold_en:
                    # English from buffer
                    if self.buffer_en:
                        text = self.buffer_en.pop(random.randint(0, len(self.buffer_en) - 1))
                    else:
                        # Buffer empty, skip
                        continue
                elif r < threshold_wiki:
                    # Italian Wiki from buffer
                    if self.buffer_it_wiki:
                        text = self.buffer_it_wiki.pop(random.randint(0, len(self.buffer_it_wiki) - 1))
                    else:
                        continue
                else:
                    # Italian Instruct (already local, just sample)
                    item = random.choice(self.buffer_it_instruct)
                    inst = item.get("instruction", "")
                    inp = item.get("input", "")
                    out = item.get("output", "")
                    full_inst = f"{inst}\n{inp}".strip()
                    text = self._format_chatml(full_inst, out)
                    
            except Exception as e:
                continue
                
            if not text:
                continue
                
            # Tokenization
            try:
                tokens = self.tokenizer.encode(text)
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    tokens.append(eos_id)
                
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                    
                if len(tokens) < 10:
                    continue
                    
                batch.append(torch.tensor(tokens, dtype=torch.long))
                
                if len(batch) >= self.batch_size:
                    yield self._collate(batch)
                    batch = []
                    
            except Exception as e:
                continue

    def _collate(self, batch):
        max_len = max(len(x) for x in batch)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        
        padded_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        
        for i, x in enumerate(batch):
            padded_batch[i, :len(x)] = x
            
        return padded_batch
