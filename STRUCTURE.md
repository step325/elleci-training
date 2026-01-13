# NanoPrime - Directory Structure

```
NanoPrime/
├── README.md                    # Main documentation
├── Fine Ultimo.md               # Project vision
├── requirements.txt             # Dependencies
├── nanoprime_v2_final.pth      # Trained model (143M params)
│
├── inference_v2.py              # Production inference script
│
├── src/                         # Core architecture (FROZEN)
│   ├── config.py                # Configuration classes
│   ├── model.py                 # NanoPrime main model
│   └── modules/                 # Components
│       ├── bitnet.py            # BitLinear layers
│       ├── mamba.py             # Mamba SSM
│       ├── mla.py               # Multi-Head Latent Attention
│       ├── router.py            # Adaptive router (experimental)
│       └── thinking_loop.py     # Recurrent loop
│
├── data/                        # Dataset loaders
│   ├── tinystories.py           # Synthetic TinyStories
│   └── __init__.py
│
├── training/                    # Training scripts
│   ├── train.py                 # Main training (v2)
│   └── __init__.py
│
├── tests/                       # Unit tests
│   ├── test_bitnet.py
│   └── __init__.py
│
├── scripts/                     # Utilities
│   └── validate_setup.py
│
├── checkpoints/                 # Empty (cleaned)
└── logs/                        # Empty (cleaned)
```

## Clean Repository Checklist

**Kept (Production)**:
- ✅ Core architecture (`src/`)
- ✅ Training pipeline (`training/train.py`)
- ✅ Final model (`nanoprime_v2_final.pth`)
- ✅ Production inference (`inference_v2.py`)
- ✅ Documentation (`README.md`, `Fine Ultimo.md`)

**Removed (Debug/Obsolete)**:
- ✅ `debug_*.py` (all debug scripts)
- ✅ `compare_v2_v3.py` (comparison utility)
- ✅ `analyze_generations.py` (analysis script)
- ✅ `generate_stories.py` (replaced by `inference_v2.py`)
- ✅ Old markdown docs (TRAINING_VERSIONS, V3_FIX_SUMMARY)
- ✅ Intermediate checkpoints (kept only final)

**Status**: Repository clean and production-ready! ✅
