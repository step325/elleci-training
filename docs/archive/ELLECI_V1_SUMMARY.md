# 🦁 Elleci V1 - Project Summary

## Overview
**Elleci V1** è un LLM italiano da 1.5B parametri con architettura ibrida innovativa.

---

## 🏗️ Architettura

| Componente | Dettagli |
|------------|----------|
| **Dimensione** | 2048d x 24 layers (~1.5B params) |
| **Attenzione** | MLA (Multi-head Latent Attention) |
| **Sequenza** | Mamba (State Space Model) |
| **Quantizzazione** | BitNet 1.58-bit |
| **Context** | 1024 tokens |
| **Vocab** | 32,128 tokens (custom BPE EN+IT) |

---

## ⚡ Ottimizzazioni Training (Implementate)

### Velocità
| Tecnica | Speedup | Status |
|---------|---------|--------|
| Seq Curriculum (256→512→1024) | 17x early | ✅ |
| WSD Scheduler | 2.9x loss | ✅ |
| 8-bit AdamW | Memoria -25% | ✅ |
| Gradient Checkpointing | Memoria -40% | ✅ |
| CUDA Optimizations | 3-4x | ✅ |

### Convergenza
| Tecnica | Miglioramento | Status |
|---------|---------------|--------|
| LeRaC (per-layer LR) | 65% loss | ✅ |
| SWA (ultimo 20%) | Generalizzazione | ✅ |

---

## 📊 Dataset

### Phase 1 - Knowledge (90%)
- **55%** Cosmopedia V2 (English textbooks)
- **35%** Wikipedia Italiana
- **10%** Istruzioni IT (7,673 samples)

### Phase 2 - Alignment (10%)
- **20%** English maintenance
- **25%** IT Wiki maintenance
- **55%** Istruzioni IT

---

## 🎯 Training Config

```python
TOTAL_STEPS = 50,000
BATCH_SIZE = 16
LEARNING_RATE = 1.5e-3
GRAD_ACCUM = 4
WARMUP = 5% (2,500 steps)
COOLDOWN = 20% (10,000 steps)  # WSD Scheduler
SWA_START = 80% (40,000 steps)
PHASE_SWITCH = 90% (45,000 steps)
```

---

## 📁 File Structure

```
NanoPrime/
├── src/
│   ├── model.py          # NanoPrime model
│   ├── config.py         # Configuration
│   └── modules/
│       ├── mamba.py      # Mamba SSM
│       ├── mla.py        # Multi-head Latent Attention
│       └── bitnet.py     # 1.58-bit quantization
├── scripts/
│   └── train_chimera.py  # Main training script
├── data/
│   └── chimera_dataset.py
└── tokenizer_chimera_v2_patched/
```

---

## 🚀 Run Training

```bash
python scripts/train_chimera.py
```

Dry run:
```bash
python scripts/train_chimera.py --dry-run --steps 20
```

---

## ⏱️ Tempo Stimato

| Fase | Steps | Stima |
|------|-------|-------|
| Phase 1 (Knowledge) | 45,000 | ~4-5 giorni |
| Phase 2 (Alignment) | 5,000 | ~0.5 giorni |
| **Totale** | 50,000 | **~5-6 giorni** |

---

## 🔮 Post-Training (Futuro)

1. **RLHF Reward Model** - Costruire reward
2. **Iterative DPO** - Alignment
3. **DiffCoT** - Reasoning fine-tuning
4. **Router Activation** - Slow/Fast path

---

*Elleci V1 - Un LLM italiano efficiente e intelligente* 🇮🇹
