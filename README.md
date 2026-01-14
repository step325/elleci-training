# Elleci V2 (Elleci)

**A 1.5B parameter hybrid LLM designed for efficiency.**

## Architecture (V2-A)
Elleci V2 integrates three efficiency paradigms into a single hybrid backbone:

| Component | Choice | Why? |
|:----------|:-------|:-----|
| **Core Backbone** | **Mamba-2 (SSD)** | Linear-time sequence modeling (infinite context potential). |
| **Attention** | **MLA** (Multi-Head Latent Attention) | Compressed KV cache (RoPE based). |
| **Weights** | **BitNet 1.58b** | Ternary weights for extreme memory efficiency. |
| **FFN** | **SwiGLU** | Gated activation for better convergence (V2 Upgrade). |
| **Norm** | **RMSNorm** | Faster and more stable than LayerNorm (V2 Upgrade). |

## Experiment
**Goal**: Validate the stability and performance of the V2 hybrid architecture on a mixed Italian/English corpus.

- **Dataset**: Chimera (55% Italian Instructions / 45% English Cosmopedia).
- **Size**: 1.5 Billion Parameters.
- **Context**: 1024 tokens (Training), 128k supported (inference).
- **Optimization**: LeRaC (Layer-wise Learning Rate), Adafactor.

## Usage

### Training
```bash
# Standard Training
python scripts/train_elleci.py

# Dry Run (Test hardware/throughput)
python scripts/train_elleci.py --dry-run
```

### Checks
```bash
# Pre-training validation
python pre_training_check.py
```
