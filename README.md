# Elleci V1 - Italian/English LLM

A 1.5B parameter hybrid LLM using **Mamba-2 + MLA + BitNet** architecture.

## Quick Start (Vast.ai / 4090)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/elleci-training.git
cd elleci-training

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train
python -m scripts.train_elleci
```

## Architecture

| Component | Description |
|-----------|-------------|
| **Mamba-2 (SSD)** | State Space Model for long sequences |
| **MLA** | Multi-head Latent Attention (KV compression) |
| **BitNet 1.58b** | Ternary weights for efficiency |
| **Adafactor** | Memory-efficient optimizer |

## Configuration

- **Model**: 2048d × 24L (~1.5B params)
- **Vocab**: 32,128 tokens (Italian-focused)
- **Batch**: 8 (effective 64 with grad accum)
- **Precision**: bfloat16

## Hardware Requirements

| GPU | VRAM | Status |
|-----|------|--------|
| RTX 4090 | 24GB | ✅ Recommended |
| RTX 4070 | 12GB | ⚠️ Reduce batch_size to 2 |
| A100 | 40GB+ | ✅ Increase batch_size |

## Training

```bash
# Standard training (50k steps)
python -m scripts.train_elleci

# Dry run (test setup)
python -m scripts.train_elleci --dry-run

# With torch.compile (experimental)
python -m scripts.train_elleci --compile

# Without WandB logging
python -m scripts.train_elleci --no-wandb
```

## Files

```
elleci-training/
├── src/
│   ├── model.py          # Main model
│   ├── config.py         # Configuration
│   └── modules/          # Mamba, MLA, BitNet, etc.
├── scripts/
│   └── train_elleci.py   # Training script
├── data/
│   └── elleci_dataset.py # Dataset loader
├── tokenizer_elleci_v1/  # Tokenizer files
└── requirements.txt
```

## License

MIT
