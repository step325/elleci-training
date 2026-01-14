# 🚀 Elleci Training Optimization Log

## 🏗️ Model Architecture & Parameters
**Model Name:** Elleci v2 (Hybrid Architectures)
**Total Parameters:** ~1.5 Billion
**Sequence Length:** 1024 tokens
**Vocab Size:** 32,128 (tiktoken)

### Detailed Configuration
- **Hidden Dimension (`d_model`):** 2048
- **Layers (`n_layers`):** 24
- **Architecture Type:** Hybrid Mamba + Attention + BitNet
- **Components:**
  - **Mamba (SSM):** Used for efficient sequence modeling
  - **MLA (Multi-Latent Attention):** 16 Heads (Head Dim: 128)
  - **BitNet:** 1.58-bit Quantization aware training
  - **Router:** Mixture-of-Experts style routing (where applicable)

## 🖥️ Hardware Environment
**GPU:** NVIDIA GeForce RTX 4070 (12GB VRAM)
**OS:** Windows 11
**Framework:** PyTorch 2.7.0+cu128
**Precision:** Mixed Precision (Float16)

---

This document records all optimization attempts to reduce training time from original estimate (~70 days) to final estimate (~17 days).

## ✅ Successful Optimizations (Applied)

| Optimization | Speedup | Description |
|---|---|---|
| **Flash Attention (SDPA)** | **5.93x (attention only)** | Replaced manual attention implementation using `F.scaled_dot_product_attention`. This massive gain comes from fused kernels and memory efficiency. |
| **cudnn.benchmark** | **3.16x** | Enabled `torch.backends.cudnn.benchmark = True`. Allows CuDNN to auto-tune convolution algorithms for the specific hardware/input size. |
| **TF32 MatMul** | **~1.0x (quality)** | Enabled `allow_tf32`. While speedup was minimal on small tests, it enables Tensor Cores on Ampere GPUs for significantly faster FP32-like math. |
| **Anomaly Detection OFF** | **4.01x (overhead)** | Explicitly disabled `autograd.set_detect_anomaly(False)`. This is a debugging tool that adds massive overhead if left on by mistake. |
| **Memory Contiguous** | **4.37x (micro)** | Ensured tensors are contiguous in memory before operations where possible. |
| **Batch Size 16** | **2.09x** | Confirmed that Batch Size 16 is optimal (11,905 tok/s). Smaller batches underutilize GPU; larger batches cause VRAM swap/OOM. |
| **Config Fix (16 Heads)** | **Stability** | Fixed bug where `d_model=2048` with `12 heads` caused invalid shapes. Updated to use `16 heads` (head_dim=128). |

---

## ⚠️ Tested & Rejected (Partial Success / Risky)

| Optimization | Result | Reason for Rejection |
|---|---|---|
| **Disable Gradient Checkpointing** | **1.31x Speedup** | **REJECTED.** Requires ~7.3GB VRAM for a tiny model. For the full 1.5B model, it exceeds 12GB VRAM, causing OOM or severe swapping. Safe training requires checkpointing on this hardware. |
| **WSL2 Environment** | **Same Performance** | SDPA speedup was similar (5.08x vs 5.93x on Windows). However, `torch.compile` failed due to `nvcc` permission issues on mounted filesystem. |

---

## ❌ Failed Experiments

| Experiment | Error | Cause |
|---|---|---|
| **torch.compile (Windows)** | `RuntimeError` | Triton compiler is not supported natively on Windows yet. |
| **torch.compile (WSL2)** | `PermissionError: nvcc` | WSL2 could not access `nvcc` correctly on the mounted Windows filesystem (`/mnt/c/...`). |
| **CUDA Graphs** | `RuntimeError` | The model/optimizer graph is too complex for simple capture (streams dependency error). Requires significant architectural rewrite to be graph-compatible. |
| **Stochastic Depth** | `AttributeError/ShapeMismatch` | Implementation clashed with Elleci's specific block structure. While theoretically sound, debugging it introduced too much risk of altering model behavior. |
| **Full Model Memory Test** | `OOM (0 bytes free)` | 12GB VRAM is extremely tight for 1.5B parameters. Even with 8-bit Optimizer and FP16, Windows memory fragmentation prevented allocation of the full contiguous model without checkpointing. |

---

## 📊 Final Status

**Original Estimate:** ~70 Days (31s/step)
**Optimized Estimate:** ~17 Days (7.5s/step)
**Total Acceleration:** **~4.1x**

### Final Optimized Configuration
- **Precision:** Mixed Precision (Float16)
- **Optimizer:** `bitsandbytes` 8-bit AdamW
- **Attention:** SDPA (Flash Attention)
- **Compiling:** Disabled (due to Windows/Triton limits)
- **Gradient Checkpointing:** **ENABLED** (Critical for 12GB RAM)
- **Batch Size:** 16
- **System Tweaks:** `cudnn.benchmark=True`, `TF32=True`

---

## 💡 Recommendations for Future Speedups
If 17 days is still too long, the only remaining options require hardware changes or model sacrifices:
1. **Reduce Sequence Length:** Cutting context from 1024 to 512 would double speed (~8-9 days).
2. **Cloud GPU:** Renting an A100 (80GB) would allow disabling checkpointing and increasing batch size massive (likely to ~3-4 days).
3. **Linux Dual Boot:** A native Linux install *might* fix `torch.compile` issues, potentially giving another 1.2-1.5x speedup.
