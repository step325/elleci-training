# Elleci V2 - Post-Training Research

Raccolta di paper e tecniche per migliorare Elleci dopo il pre-training.

---

## 🎯 ALIGNMENT & PREFERENCE LEARNING

### Iterative DPO
- **Link**: https://arxiv.org/abs/2312.11456
- **Cosa fa**: DPO iterativo con esplorazione online del policy space
- **Come migliora Elleci**: Supera DPO standard del 5-10% su allineamento. Permette di raffinare iterativamente le preferenze invece di un singolo round.
- **Difficoltà**: ⭐⭐⭐ (Media)

### RLAIF vs RLHF
- **Link**: https://arxiv.org/abs/2309.00267
- **Cosa fa**: Usa un LLM (anche lo stesso!) per generare preferenze invece di annotatori umani
- **Come migliora Elleci**: **Self-improvement senza costi di annotazione!** Elleci può valutare le proprie risposte e migliorarsi.
- **Difficoltà**: ⭐⭐ (Facile)

### RLAIF-V (Self-Alignment)
- **Link**: https://arxiv.org/abs/2405.17220
- **Cosa fa**: Self-feedback per ridurre allucinazioni (-80%)
- **Come migliora Elleci**: Riduzione drastica delle allucinazioni senza dati esterni. Il modello impara dai propri errori.
- **Difficoltà**: ⭐⭐⭐ (Media)

### UNA (Unified Natural Alignment)
- **Link**: https://arxiv.org/abs/2408.15339
- **Cosa fa**: Allineamento da feedback scalare (like/dislike) invece di preferenze complesse
- **Come migliora Elleci**: Feedback più semplice da raccogliere dagli utenti. Basta un thumbs up/down.
- **Difficoltà**: ⭐⭐ (Facile)

---

## 🧠 REASONING & SELF-IMPROVEMENT

### SPAG (Self-Playing Adversarial Game)
- **Link**: https://arxiv.org/abs/2404.10642
- **Cosa fa**: Due copie del modello giocano un gioco adversarial (Taboo) per migliorare il reasoning
- **Come migliora Elleci**: **Migliora reasoning senza dati esterni!** Solo self-play iterativo.
- **Difficoltà**: ⭐⭐⭐ (Media)

### DiffCoT (Diffusion Chain-of-Thought)
- **Link**: https://arxiv.org/abs/2601.03559
- **Cosa fa**: CoT come processo di denoising diffusion con auto-correzione
- **Come migliora Elleci**: Errori nei primi step non propagano irreversibilmente. Il modello può "tornare indietro" e correggere.
- **Difficoltà**: ⭐⭐⭐⭐ (Difficile)

### Long CoT Survey
- **Link**: https://arxiv.org/abs/2503.09567
- **Cosa fa**: Survey completa su Long CoT: deep reasoning + exploration + reflection
- **Come migliora Elleci**: Framework teorico per implementare reasoning complesso stile o1/DeepSeek-R1.
- **Difficoltà**: ⭐⭐⭐⭐ (Difficile, richiede architettura specifica)

---

## 🔀 MODEL MERGING & CRESCITA

### Dataless Knowledge Fusion
- **Link**: https://arxiv.org/abs/2212.09849
- **Cosa fa**: Merge di modelli nello spazio dei pesi senza accesso ai dati di training
- **Come migliora Elleci**: Combina Elleci con modelli specialisti (code, math) senza retraining.
- **Difficoltà**: ⭐ (Molto facile, solo merge pesi)

### FSLoRA (Federated Sketching LoRA)
- **Link**: https://arxiv.org/abs/2501.19389
- **Cosa fa**: LoRA con sketching per adattare a risorse eterogenee
- **Come migliora Elleci**: Fine-tuning efficiente su hardware limitato.
- **Difficoltà**: ⭐⭐ (Facile)

### TLI (Targeted Lexical Injection)
- **Link**: https://arxiv.org/abs/2506.15415
- **Cosa fa**: LoRA su early layers per alignment cross-lingue
- **Come migliora Elleci**: Migliora allineamento italiano-inglese (+28% similarity).
- **Difficoltà**: ⭐⭐ (Facile)

---

## 📊 DATA & DISTILLATION

### LLM Synthetic Data Survey
- **Link**: https://arxiv.org/abs/2406.15126
- **Cosa fa**: Survey completa su generazione dati sintetici con LLM
- **Come migliora Elleci**: Framework per generare dati di training di qualità usando LLM più grandi.
- **Difficoltà**: ⭐⭐ (Facile)

### Vision-Flan
- **Link**: https://arxiv.org/abs/2402.11690
- **Cosa fa**: Instruction tuning in 2 fasi: task diversi prima, GPT-4 data dopo
- **Come migliora Elleci**: Solo ~1000 sample GPT-4 servono per allineare le risposte! Task diversity > quantity.
- **Difficoltà**: ⭐⭐ (Facile)

### LESS (Data Selection)
- **Link**: https://arxiv.org/abs/2402.04333
- **Cosa fa**: Seleziona i dati di training più informativi
- **Come migliora Elleci**: Riduce dataset necessario del 50-90% mantenendo performance.
- **Difficoltà**: ⭐⭐⭐ (Media)

### Corpus Distillation Framework
- **Link**: https://arxiv.org/abs/2504.19565
- **Cosa fa**: Multi-agent per estrarre Q&A da letteratura scientifica
- **Come migliora Elleci**: Genera dataset domain-specific di alta qualità automaticamente.
- **Difficoltà**: ⭐⭐⭐ (Media)

---

## 🚀 FASE 2: PRIORITÀ IMPLEMENTAZIONE

### Alta Priorità (Facili, Alto Impatto)

| # | Tecnica | Tempo | Impatto |
|---|---------|-------|---------|
| 1 | **RLAIF** (self-reward) | 1-2 giorni | 🔥🔥🔥🔥 |
| 2 | **Dataless Knowledge Fusion** | 2-3 ore | 🔥🔥🔥 |
| 3 | **UNA** (feedback scalare) | 1 giorno | 🔥🔥🔥 |
| 4 | **FSLoRA/LoRA** fine-tuning | 1 giorno | 🔥🔥🔥 |

### Media Priorità (Più Complesse)

| # | Tecnica | Tempo | Impatto |
|---|---------|-------|---------|
| 5 | **SPAG** (self-play reasoning) | 3-5 giorni | 🔥🔥🔥🔥 |
| 6 | **Iterative DPO** | 2-3 giorni | 🔥🔥🔥 |
| 7 | **RLAIF-V** (anti-hallucination) | 2-3 giorni | 🔥🔥🔥 |
| 8 | **TLI** (IT-EN alignment) | 1 giorno | 🔥🔥 |

### Avanzate (Richiedono R&D)

| # | Tecnica | Tempo | Impatto |
|---|---------|-------|---------|
| 9 | **DiffCoT** | 1-2 settimane | 🔥🔥🔥🔥🔥 |
| 10 | **Long CoT** (o1-style) | 2-4 settimane | 🔥🔥🔥🔥🔥 |

---

## 📋 Roadmap Suggerita

```
┌─────────────────────────────────────────────────────────────┐
│                    ELLECI V1 (Pre-training)                 │
│                         ✅ IN CORSO                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASE 2A: SFT Base                        │
│  • OpenOrca/Alpaca-CoT instruction tuning                   │
│  • LoRA fine-tuning (efficiente)                           │
│  • Tempo: ~1-2 giorni                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASE 2B: Self-Improvement                │
│  • RLAIF (self-reward, nessun annotatore)                  │
│  • SPAG (self-play per reasoning)                          │
│  • Tempo: ~3-5 giorni                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASE 2C: Specializzazione                │
│  • Model merging con specialisti (code, math)              │
│  • DiffCoT per reasoning avanzato                          │
│  • Tempo: ~1-2 settimane                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ELLECI V2 🚀                             │
│  • Reasoning migliorato                                     │
│  • Self-improvement continuo                                │
│  • Allucinazioni ridotte                                    │
└─────────────────────────────────────────────────────────────┘
```

---

*Documento generato: 2026-01-13*
