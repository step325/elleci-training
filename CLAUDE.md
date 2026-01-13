# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NanoPrime/Elleci** is a hybrid language model project demonstrating extreme efficiency through architectural innovation. The goal is to create a "Pocket Genius" - an LLM that runs on consumer hardware (RTX 4070) with deep reasoning capabilities.

### Core Architecture
- **142.6M parameters** (v2) scaling to **1.5B parameters** (Elleci v1)
- **Hybrid components**: BitNet (1.58-bit quantization) + Mamba (SSM) + MLA (Multi-Head Latent Attention)
- **Novel features**: Alternating Mamba/MLA layers for O(N) speed with compressed attention (93% KV cache reduction)
- **Target**: System 2 reasoning with thinking tokens `<think>...</think>`

## Commands

### Training
```bash
# Main training (TinyStories synthetic - v2)
python training/train.py

# Chimera training (1.5B params, Italian/English mixed)
python scripts/train_chimera.py

# Dry run (testing)
python scripts/train_chimera.py --dry-run --steps 20

# Cosmopedia training (future real dataset)
python training/train_cosmopedia.py
```

### Inference
```bash
# Interactive generation
python inference_v2.py

# Batch generation
python inference_v2.py --batch 5

# API server (FastAPI)
python serve.py
# Then access at http://localhost:8000
```

### Web Interface
```bash
# Frontend (Next.js)
cd web_chat
npm install
npm run dev  # Development at http://localhost:3000
npm run build && npm start  # Production
```

### Testing
```bash
# Run unit tests
pytest tests/

# Specific test
pytest tests/test_bitnet.py
```

### Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Database connection (Railway PostgreSQL) - READ ONLY
PGPASSWORD=YyXSgLUTnDxOAcalJEDGXPHxXbPzkVhE psql -h maglev.proxy.rlwy.net -U postgres -p 18523 -d railway
```

## Architecture Details

### Module Structure
```
src/
├── config.py          # Centralized configuration (dataclasses)
├── model.py           # NanoPrime main model (hybrid layers)
└── modules/
    ├── bitnet.py      # BitLinear 1.58-bit quantization layers
    ├── mamba.py       # Mamba v1 SSM (state space model)
    ├── mamba2.py      # Mamba v2 SSD (structured state dilation)
    ├── mla.py         # Multi-Head Latent Attention (compressed KV)
    ├── router.py      # Adaptive routing (fast/slow paths - experimental)
    ├── thinking_loop.py  # Recurrent reasoning loop
    └── pscan.py       # Parallel scan primitives
```

### Key Innovations
1. **BitNet 1.58b**: Weights quantized to {-1, 0, +1} with learned scaling. 16x memory reduction vs FP16.
2. **Mamba SSM**: O(N) complexity for long sequences, replaces quadratic attention in alternating layers.
3. **MLA**: Compresses Key/Value to low-rank (kv_lora_rank=128), then expands. 93% KV cache reduction.
4. **Hybrid Alternation**: Layers alternate between Mamba (long-range) and MLA (local precision).

### Configuration System
All configs are dataclasses in `src/config.py`:
- `NanoPrimeConfig`: Main model (d_model=768, n_layers=6-24, vocab_size=50304)
- `BitNetConfig`: Quantization (eps=1e-5)
- `MLAConfig`: Attention (n_heads=12, kv_lora_rank=128, rope_dim=32)
- `MambaConfig`: SSM (d_state=16, d_conv=4, use_mamba2=True)
- `RouterConfig`: Gating (n_experts=2, temperature=1.0)
- `ThinkingLoopConfig`: Reasoning (max_iterations=4)

To modify architecture, edit config in training scripts or create new config instances.

### Training Pipeline (Chimera v1)
**Two-phase approach**:
1. **Phase 1 (90% - 45K steps)**: Knowledge acquisition
   - 55% Cosmopedia V2 (English textbooks)
   - 35% Wikipedia Italiana
   - 10% Italian instructions (7,673 samples)
2. **Phase 2 (10% - 5K steps)**: Instruction alignment
   - 20% English maintenance
   - 25% IT Wiki maintenance
   - 55% Italian instructions

**Key optimizations**:
- Sequence curriculum (256→512→1024 tokens): 17x early speedup
- WSD Scheduler (warmup-stable-decay): 2.9x loss improvement
- 8-bit AdamW (bitsandbytes): 25% memory reduction
- Gradient checkpointing: 40% memory reduction
- LeRaC (per-layer learning rates): 65% loss improvement
- SWA (Stochastic Weight Averaging): Last 20% of training

### Datasets
- **TinyStories**: Synthetic children's stories (~13 tokens avg) - used in v2 proof-of-concept
- **Cosmopedia V2**: 30M educational textbooks (future Phase 1)
- **Wikipedia IT**: Italian language knowledge
- **Instructions**: Located in `data/elleci_instructions.jsonl` and `data/chimera_instructions_final.jsonl`

Custom dataset loaders in `data/` directory:
- `tinystories.py`: Streaming TinyStories (HuggingFace)
- `elleci_dataset.py`: Mixed Italian/English corpus with phase switching

## Project Status

### Current State (v2)
- ✅ 142.6M parameter hybrid model functional
- ✅ Trained on TinyStories (10K steps, 52min, loss=0.338)
- ✅ All efficiency techniques integrated and working
- ✅ Production inference script ready
- ⚠️ Generation is coherent but repetitive (synthetic data limitation)

### Roadmap
1. **Phase 1 - Data Quality** (CRITICAL): Replace TinyStories with Cosmopedia + OpenWebMath + The Stack
2. **Phase 2 - Longer Training**: 50K-100K steps (Mamba needs more steps)
3. **Phase 3 - System 2 Reasoning**: Add Chain-of-Thought data with thinking tokens
4. **Phase 4 - Scale & Deploy**: 280M params + INT8 quantization

### Known Files
- `nanoprime_v2_final.pth`: Current trained checkpoint (142.6M params, 6 layers)
- `nanoprime_v2_italian.pth`: Italian fine-tuned version (referenced in serve.py)
- `tokenizer_chimera_v2_patched/`: Custom BPE tokenizer (32,128 vocab, EN+IT)

## Special Considerations

### CUDA Optimizations
Training scripts use aggressive CUDA optimizations (see `scripts/train_chimera.py`):
```python
torch.backends.cudnn.benchmark = True  # 3.16x speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(False)  # 4x speedup
```
These are benchmark-verified on RTX 4070.

### Tokenizer
- GPT-2 tokenizer (50,304 vocab) for English/v2
- Custom Chimera tokenizer (32,128 vocab) in `tokenizer_chimera_v2_patched/` for Italian
- Always use `tokenizer.pad_token = tokenizer.eos_token`

### Device Management
Models auto-detect CUDA availability via `config.device`. Default fallback is CPU (very slow).

### Model Loading Pattern
```python
from src.config import NanoPrimeConfig
from src.model import NanoPrime

config = NanoPrimeConfig()
config.n_layers = 6  # Match checkpoint
config.max_seq_len = 64  # Match training
model = NanoPrime(config)
model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
model.eval()
```

### Web Stack
- **Backend**: FastAPI (serve.py) with CORS enabled for local dev
- **Frontend**: Next.js 16 with React 19, TypeScript, TailwindCSS 4
- **Communication**: REST API at http://localhost:8000

## Development Workflow

1. **Architecture changes**: Edit `src/config.py` and corresponding module in `src/modules/`
2. **Training experiments**: Copy and modify `training/train.py` or use `scripts/train_chimera.py`
3. **New datasets**: Create loader in `data/` following `tinystories.py` pattern
4. **Testing**: Add tests in `tests/` and run with pytest
5. **Deployment**: Use `inference_v2.py` for CLI or `serve.py` + `web_chat` for web interface

## References

- Vision document: `Fine Ultimo.md`
- Project summary: `ELLECI_V1_SUMMARY.md`
- Structure overview: `STRUCTURE.md`
- Optimization log: `OPTIMIZATION_LOG.md`
