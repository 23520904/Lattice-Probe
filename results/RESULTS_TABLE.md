# LatticeProbe — Results Recording Table

Fill in each section after every run. Keep all runs — don't overwrite old entries.

---

## How to read a run entry

- **N** = number of training samples (e.g. 2^16 = 65 536)
- **AUROC** = mean over 100 bootstrap resamples
- **95% CI** = [lo, hi] from `evaluate.py`
- **Epochs** = actual epochs run (may be less than 50 if early-stopping triggered)
- **GPU-h** = total GPU-hours from `compute_log.csv` for that run

---

## Table 1 — Sanity checks (must pass before standardised results are meaningful)

| Param set | Model | N | AUROC | 95% CI | Epochs | GPU-h | Paper target | Pass? |
|---|---|---|---|---|---|---|---|---|
| W2 (no noise) | Transformer | | | [ , ] | | | ≈ 1.000 | |
| W2 (no noise) | GNN | | | [ , ] | | | ≈ 1.000 | |
| W1 (binary secret) | Transformer | | | [ , ] | | | ≥ 0.710 | |
| W1 (binary secret) | GNN | | | [ , ] | | | ≥ 0.710 | |
| W3 (σ −60%) | Transformer | | | [ , ] | | | ≥ 0.750 | |
| W3 (σ −60%) | GNN | | | [ , ] | | | ≥ 0.750 | |

> ⚠️ If W2 AUROC < 0.99 the pipeline has a bug. Do not record standardised results until all three pass.

---

## Table 2 — Standardised parameters (paper Table 2)

### Transformer

| Param set | N | AUROC | 95% CI | Epochs | GPU-h | Paper AUROC |
|---|---|---|---|---|---|---|
| ML-KEM-512 | | | [ , ] | | | 0.502 |
| ML-KEM-512 | | | [ , ] | | | 0.502 |
| ML-KEM-768 | | | [ , ] | | | 0.500 |
| ML-KEM-1024 | | | [ , ] | | | 0.500 |

### GNN

| Param set | N | AUROC | 95% CI | Epochs | GPU-h | Paper AUROC |
|---|---|---|---|---|---|---|
| ML-KEM-512 | | | [ , ] | | | 0.501 |
| ML-KEM-768 | | | [ , ] | | | 0.500 |
| ML-KEM-1024 | | | [ , ] | | | 0.500 |

### χ² statistical baseline

| Param set | N | AUROC | 95% CI | Paper AUROC |
|---|---|---|---|---|
| ML-KEM-512 | | | [ , ] | 0.500 |
| ML-KEM-768 | | | [ , ] | 0.500 |
| ML-KEM-1024 | | | [ , ] | 0.500 |

---

## Table 3 — Edge-of-margin (GNN only)

| N | AUROC | 95% CI | Epochs | GPU-h | Paper AUROC |
|---|---|---|---|---|---|
| 2^14 = 16 384 | | | [ , ] | | — |
| 2^16 = 65 536 | | | [ , ] | | — |
| 2^18 = 262 144 | | | [ , ] | | 0.541 |

---

## Table 4 — Partial-bit secret recovery

| Param set | N (train) | Bits tested | Mean accuracy | Random baseline | Pass? |
|---|---|---|---|---|---|
| ML-KEM-512 | | 512 | | 0.500 | |
| ML-KEM-768 | | 768 | | 0.500 | |
| ML-KEM-1024 | | 1024 | | 0.500 | |

> Pass = mean accuracy within ±0.01 of 0.500

---

## Table 5 — Compute scaling (ML-KEM-512, GNN)

| N | Val AUROC | GPU-h | Notes |
|---|---|---|---|
| 2^14 = 16 384 | | | |
| 2^16 = 65 536 | | | |
| 2^18 = 262 144 | | | |
| 2^20 = 1 048 576 | | | |

---

## Run log

Record each training run here so you can track what was done.

| # | Date | Param set | Model | N | Epochs | Best val AUROC | Notes |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |
| 9 | | | | | | | |
| 10 | | | | | | | |

---

## How to fill in after a run

After `evaluate.py` finishes, the output looks like:

```
===================================================
Param set : ML-KEM-512   Model : transformer   N_test : 32768
===================================================
  transformer (ours)             AUROC = 0.5021  95% CI [0.4987, 0.5055]
  χ² statistical                 AUROC = 0.5001  95% CI [0.4970, 0.5033]
===================================================
```

Copy AUROC and CI directly into the table above.
GPU-hours come from `compute_log.csv` — sum the `gpu_hours` column for all epochs of that run.
