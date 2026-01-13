# Chimera Dataset Documentation

## Overview

The **ChimeraDataset** is a streaming dataset for training bilingual (English-Italian) language models with direct streaming from HuggingFace (no local buffering).

## Data Sources

### 1. English: Cosmopedia V2
- **Source**: `HuggingFaceTB/smollm-corpus` (cosmopedia-v2)
- **Size**: Large educational content corpus
- **Purpose**: English language knowledge and reasoning

### 2. Italian: CulturaX
- **Source**: `uonlp/CulturaX` (it subset)
- **Size**: 200GB+ cleaned Italian text (41 billion words)
- **Quality**: Highest quality Italian corpus available (deeply cleaned and deduplicated)
- **Fallback**: Wikipedia IT if CulturaX fails to load

### 3. Italian Instructions
- **Source**: Local JSONL files
- **Files**:
  - `chimera_instructions_final.jsonl` (7,672 samples)
  - `elleci_instructions.jsonl` (164 samples)
- **Total**: 7,836 instruction-response pairs
- **Format**: ChatML with `<|im_start|>` tags

## Training Phases

### Phase 1: Knowledge Acquisition (90% of training, ~45K steps)
- **55%** English Cosmopedia V2 (educational content)
- **35%** Italian CulturaX (general knowledge)
- **10%** Italian Instructions (task alignment)

**Purpose**: Build broad knowledge base in both languages

### Phase 2: Instruction Alignment (10% of training, ~5K steps)
- **20%** English Cosmopedia V2 (maintenance)
- **25%** Italian CulturaX (maintenance)
- **55%** Italian Instructions (heavy alignment)

**Purpose**: Align model to instruction-following behavior

## Features

### Direct Streaming
- No local buffer (memory efficient)
- Infinite data stream (prevents overfitting)
- Dynamic shuffling (buffer_size=1000 on HuggingFace side)

### Dynamic Batching
- Sequences padded to max length in batch
- Variable batch sizes (configured at init)
- EOS token automatically appended

### Sequence Length Curriculum
The training script (`train_chimera.py`) progressively increases sequence length:
- Steps 0-20K: 256 tokens (fast, 17x speedup)
- Steps 20K-35K: 512 tokens
- Steps 35K-50K: 1024 tokens (full context)

## Testing

### Quick Test (2-3 minutes)
```bash
python test_chimera_dataset.py
```
- Tests basic functionality
- Fetches 5 batches
- Verifies tokenization and shapes

### Comprehensive Test (5-10 minutes)
```bash
python test_chimera_comprehensive.py
```
- Tests both Phase 1 and Phase 2
- Fetches 10 batches
- Verifies token ranges and proportions
- Shows detailed statistics

## Usage in Training

```python
from transformers import PreTrainedTokenizerFast
from data.chimera_dataset import ChimeraDataset
from torch.utils.data import DataLoader

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer_chimera_v2_patched/tokenizer.json")
tokenizer.pad_token = "<|endoftext|>"

# Create dataset
dataset = ChimeraDataset(
    tokenizer=tokenizer,
    phase=1,           # or 2
    max_length=512,    # dynamic in training
    batch_size=16
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=None,  # dataset handles batching
    num_workers=0,    # streaming doesn't benefit from workers
    pin_memory=True
)

# Training loop
for batch in dataloader:
    # batch shape: [batch_size, seq_len]
    # Already padded and ready for model
    ...
```

## Troubleshooting

### Issue: CulturaX download is slow
- **First run**: Initial download can take 5-10 minutes (downloading metadata and first chunks)
- **Subsequent runs**: Cached, much faster
- **Alternative**: Will fallback to Wikipedia IT automatically

### Issue: Tokens exceed vocab_size
- Check that tokenizer vocab_size matches model config
- Run: `python test_chimera_comprehensive.py` to verify

### Issue: Out of memory
- Reduce `max_length` (e.g., 256 instead of 512)
- Reduce `batch_size` (e.g., 8 instead of 16)
- Enable gradient checkpointing in model

### Issue: Streaming hangs
- Check internet connection (required for HuggingFace)
- Verify HuggingFace token if datasets are gated
- Try fallback: manually set Wikipedia in `_get_it_stream()`

## Performance

- **Streaming speed**: ~2-3 seconds per batch (first batches slower due to download)
- **Memory usage**: Minimal (no buffer)
- **Diversity**: Infinite (no repetition)

## References

- [CulturaX Paper](https://aclanthology.org/2024.lrec-main.377.pdf)
- [CulturaX on HuggingFace](https://huggingface.co/datasets/uonlp/CulturaX)
- [Cosmopedia V2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
