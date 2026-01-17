#!/bin/bash
# ============================================================
# Elleci - Setup per vast.ai (RTX 4090)
# ============================================================

echo "========================================"
echo "  ELLECI - Setup vast.ai"
echo "========================================"

# Verifica GPU
echo ""
echo "[1/5] Verifica GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Verifica PyTorch (già nel template)
echo "[2/5] Verifica PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA {torch.cuda.is_available()}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[!] PyTorch non trovato - qualcosa non va col template vast.ai"
    exit 1
fi

# Installa pacchetti base e build tools
echo ""
echo "[3/5] Installazione dipendenze e Kernel Ottimizzati..."

# 1. Build tools (essenziali per compilare flash-attn)
# Limit jobs to prevent OOM and define local TMPDIR
# 4 jobs is safe for 64GB RAM (~16GB/job)
export MAX_JOBS=4
export TMPDIR=~/tmp_build
mkdir -p $TMPDIR
pip install --quiet packaging numpy

# 2. PyTorch dependencies base
pip install --quiet --upgrade-strategy only-if-needed \
    bitsandbytes>=0.41.0 \
    datasets>=2.14.0 \
    einops>=0.7.0 \
    wandb>=0.15.0 \
    tqdm>=4.65.0 \
    transformers>=4.35.0 \
    tokenizers>=0.15.0

# 3. Flash Attention 2 (Critico per A100)
echo "   Installazione Flash Attention 2 (può richiedere qualche minuto - modalità bilanciata 4 core)..."
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# 4. Mamba Kernels (Critico per velocità Mamba)
echo "   Installazione Mamba Kernels..."
MAX_JOBS=4 pip install causal-conv1d>=1.2.0 --no-build-isolation
MAX_JOBS=4 pip install mamba-ssm>=1.2.0 --no-build-isolation

echo "   Dipendenze e Kernel OK"

# Verifica imports critici
echo ""
echo "[4/5] Verifica imports..."
python -c "
import torch
import bitsandbytes
import datasets
import einops
from accelerated_scan.scalar import scan
print('   Tutti gli import OK')
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "[!] Alcuni import falliti - controlla errori sopra"
fi

# Verifica checkpoint
echo ""
echo "[5/5] Verifica checkpoint..."
if [ -f "checkpoints/elleci_v1_final.pth" ]; then
    SIZE=$(du -h checkpoints/elleci_v1_final.pth | cut -f1)
    echo "   Checkpoint trovato: $SIZE"
else
    echo "[!] Checkpoint NON trovato!"
    echo "    Copia il checkpoint con:"
    echo "    scp checkpoints/elleci_v1_final.pth vastai:~/Elleci/checkpoints/"
fi

echo ""
echo "========================================"
echo "  Setup completato!"
echo ""
echo "  Per avviare il fine-tuning:"
echo "  python scripts/finetune_instructions.py \\"
echo "      --checkpoint checkpoints/elleci_v1_final.pth"
echo "========================================"
