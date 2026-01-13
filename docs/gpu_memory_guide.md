# Come funziona la memoria nel training di un LLM

Una guida completa su cosa consuma memoria durante il training di un modello come Elleci (1.5B parametri).

---

## 1. Cosa occupa la VRAM (in ordine di grandezza)

```
╔═══════════════════════════════════════════════════════════════╗
║                    VRAM USAGE BREAKDOWN                       ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────────────────────────────┐                      ║
║  │     MODEL WEIGHTS (~5 GB)           │  ← Pesi permanenti   ║
║  │     1.5B params × 32bit = 6GB       │    BF16 = ~3GB       ║
║  │     (ma con embedding grande)       │                      ║
║  └─────────────────────────────────────┘                      ║
║                                                               ║
║  ┌─────────────────────────────────────┐                      ║
║  │     GRADIENTS (~5 GB)               │  ← Stessa size dei   ║
║  │     ∂Loss/∂weight per ogni peso     │    pesi (FP32)       ║
║  └─────────────────────────────────────┘                      ║
║                                                               ║
║  ┌─────────────────────────────────────┐                      ║
║  │     OPTIMIZER STATES (~2.5 GB)      │  ← AdamW mantiene    ║
║  │     - m (momentum): 1 tensor/peso   │    2 copie!          ║
║  │     - v (variance): 1 tensor/peso   │    8-bit = risparmia ║
║  └─────────────────────────────────────┘                      ║
║                                                               ║
║  ┌─────────────────────────────────────┐                      ║
║  │     ACTIVATIONS (~0.1-2 GB)         │  ← Output di ogni    ║
║  │     batch × seq × d_model × layers  │    layer (per backwd)║
║  │     (Con checkpointing = molto meno)│                      ║
║  └─────────────────────────────────────┘                      ║
║                                                               ║
║  ┌─────────────────────────────────────┐                      ║
║  │     PYTORCH CACHE (~1-3 GB)         │  ← Tensori temporanei║
║  │     Allocazioni intermedie          │    non liberati      ║
║  └─────────────────────────────────────┘                      ║
╚═══════════════════════════════════════════════════════════════╝
```

### Riepilogo per 1.5B params su 12GB VRAM:

| Componente | Size | Note |
|------------|------|------|
| Pesi modello | ~5 GB | BF16 mixed precision |
| Gradienti | ~5 GB | FP32 per stabilità |
| Optimizer (8-bit) | ~2.5 GB | Con bitsandbytes |
| Attivazioni | ~0.1 GB | Con gradient checkpointing |
| **TOTALE** | **~12.6 GB** | Appena sotto il limite! |

---

## 2. Cosa succede in un singolo training step

### STEP 1: FORWARD PASS

```
Input: [batch=4, seq=256] token IDs

  ┌──────────┐
  │ Embedding │ → Alloca [4, 256, 2048] = 8MB
  └────┬─────┘
       ↓
  ┌──────────┐
  │  Layer 1  │ → Input salvato per backward (con checkpointing: scartato)
  │  Mamba/MLA│ → Output [4, 256, 2048], tensori intermedi (~50-200MB)
  └────┬─────┘
       ↓
      ...      (× 24 layers)
       ↓
  ┌──────────┐
  │  LM Head  │ → [4, 256, 32128] = 127MB (GROSSO!)
  └────┬─────┘
       ↓
  ┌──────────┐
  │   Loss    │ → Scalar + grafo computazionale in memoria
  └──────────┘
```

### STEP 2: BACKWARD PASS

PyTorch attraversa il grafo al contrario:

```
Loss.backward() →
  │
  ├─→ Calcola ∂Loss/∂(lm_head.weight), accumula in .grad
  │   └─→ Alloca gradiente se prima era None, oppure += se esiste
  │
  ├─→ Per ogni layer (24 → 1):
  │   │
  │   ├─→ Se checkpointing: RICALCOLA forward di questo layer
  │   │   └─→ Alloca temporaneamente le attivazioni
  │   │
  │   ├─→ Calcola gradienti per questo layer
  │   │   └─→ Mamba state: d_state × matmul → tensori intermedi
  │   │
  │   └─→ Libera grafo di questo layer (ma gradients restano!)
  │
  └─→ Tutti i .grad sono ora popolati (~5GB allocati)
```

### STEP 3: OPTIMIZER.STEP()

```
optimizer.step() →
  │
  ├─→ Per ogni parametro:
  │   │
  │   ├─→ Legge param.grad
  │   │
  │   ├─→ Aggiorna momentum: m = β1*m + (1-β1)*grad
  │   │   └─→ Se 8-bit: quantizza m su 8 bit
  │   │
  │   ├─→ Aggiorna variance: v = β2*v + (1-β2)*grad²
  │   │
  │   └─→ Aggiorna peso: param -= lr * m / (√v + ε)
  │
  └─→ Momentum e variance RESTANO in memoria (optimizer state)
```

### STEP 4: ZERO_GRAD()

```
optimizer.zero_grad() →
  │
  ├─→ set_to_none=False: param.grad.zero_()  ← Tensore resta allocato!
  │
  └─→ set_to_none=True: param.grad = None    ← Tensore LIBERATO!
```

> **IMPORTANTE**: Usa sempre `zero_grad(set_to_none=True)` per liberare memoria!

---

## 3. Il ruolo del CUDA Kernel

```
┌─────────────────────────────────────────────────────────────┐
│                        CUDA ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   CPU (Host)              │    GPU (Device)                 │
│   ──────────              │    ─────────────                │
│                           │                                 │
│   Python code             │    CUDA Cores (5888 on 4070)    │
│       ↓                   │         ↓                       │
│   PyTorch API             │    Tensor Cores (184)           │
│       ↓                   │         ↓                       │
│   CUDA Runtime   ───────────→   Kernel Execution            │
│       ↓                   │         ↓                       │
│   cuDNN/cuBLAS            │    VRAM (12GB GDDR6X)           │
│                           │                                 │
└─────────────────────────────────────────────────────────────┘
```

### Cos'è un Kernel?

Un **kernel CUDA** è una funzione che gira sulla GPU in parallelo su migliaia di thread.

**Esempio**: `torch.matmul(A, B)`

1. PyTorch chiama cuBLAS (libreria NVIDIA ottimizzata)
2. cuBLAS lancia un "kernel" sulla GPU
3. Il kernel usa i Tensor Cores per moltiplicare matrici in parallelo
4. Risultato scritto in VRAM
5. Controllo torna a Python

### Problema con VRAM piena:

Quando la VRAM è piena, CUDA alloca su **Unified Memory** (RAM di sistema):
- Trasferimenti via PCIe 4.0 (~32 GB/s) invece di GDDR6X (~500 GB/s)
- **15x più lento!**

---

## 4. Perché la memoria CRESCE durante il training

### Timeline tipica:

```
Step 0-15 (gradient accumulation):
  Gradienti: accumulati in .grad (non liberati fino a step 16)
  
Step 16 (primo optimizer.step):
  ┌─────────────────────────────────────────┐
  │ optimizer.step() alloca optimizer state │
  │ - momentum tensors: ~2.5 GB            │
  │ - variance tensors: ~2.5 GB            │
  │ (Ma con 8-bit AdamW: ~1.2 GB)          │
  └─────────────────────────────────────────┘
```

### Possibili cause di memory leak:

1. **zero_grad senza set_to_none** - Gradienti restano allocati
2. **bitsandbytes buffer interni** - L'optimizer 8-bit mantiene cache
3. **Mamba state** - Lo stato SSM potrebbe accumularsi
4. **PyTorch caching** - Tensori temporanei non liberati

---

## 5. Memoria condivisa (Unified Memory)

```
┌──────────────────────────────────────────────────────────┐
│               NVIDIA UNIFIED MEMORY                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   VRAM (12GB)          RAM (32GB)                        │
│   ───────────          ──────────                        │
│   [████████████]       [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │
│      Piena!                 Pagine GPU                   │
│                                                          │
│   Quando VRAM è piena:                                   │
│   1. CUDA crea pagine "unificate" in RAM                 │
│   2. GPU accede via PCIe (lento!)                        │
│   3. Driver migra pagine avanti/indietro                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Confronto Bandwidth:

| Tipo memoria | Bandwidth | Relativo |
|--------------|-----------|----------|
| VRAM GDDR6X | ~500 GB/s | 1x |
| DDR5 6000MT/s | ~48 GB/s | 10x più lento |
| PCIe 4.0 x16 | ~32 GB/s | 15x più lento |

---

## 6. Ottimizzazioni applicate a Elleci

1. **Gradient Checkpointing** - Ricomputa attivazioni invece di salvarle
2. **8-bit AdamW** - Riduce optimizer state del 75%
3. **BFloat16 mixed precision** - Pesi e attivazioni a 16-bit
4. **Batch size ridotto** - 4 invece di 16, con gradient accumulation
5. **set_to_none=True** - Libera gradienti dopo ogni step
6. **Periodic gc.collect()** - Garbage collection Python forzato

---

*Documento generato per il progetto Elleci V1*
