# LatticeProbe

Open-source implementation of the ML-cryptanalysis framework from:

> Ologunde, E. *"AI-Accelerated Cryptanalysis of Lattice-Based Schemes: A Stress-Test of NIST PQC Parameter Choices via Transformer and Graph Neural Network Distinguishers."* 2026.

The paper trains Transformer and GraphSAGE models to distinguish Module-LWE samples from uniform random — a hard problem whose hardness underpins CRYSTALS-Kyber (ML-KEM, FIPS-203). The central finding is that **standardised ML-KEM parameters (512/768/1024) show no exploitable structure** (AUROC ≈ 0.50) even with up to 2²⁰ training samples and ~51M-parameter models, while intentionally weakened regimes are cleanly detected.

---

## Installation

**Requires Python ≥ 3.10 and PyTorch ≥ 2.12.**

```bash
git clone https://github.com/23520904/Lattice-Probe.git
cd latticeprobe
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -e .
```

To verify the install:

```bash
pytest tests/ -q
# 62 passed
```

---

## Quick start

### Generate LWE samples

```python
from latticeprobe.params import get_params
from latticeprobe.sampler import generate_lwe_sample, generate_uniform_sample, generate_batch

params = get_params("ML-KEM-512")

# Single LWE sample: b = <a, s> + e  mod q
a, b, s = generate_lwe_sample(params)
print(a.shape, b.shape, s.shape)   # (2, 256)  (256,)  (2, 256)

# Single uniform sample: b has no algebraic relation to a
a_u, b_u = generate_uniform_sample(params)

# Balanced mixed batch (50% LWE / 50% uniform)
A, B, labels = generate_batch(params, n_samples=1024)
print(A.shape, B.shape, labels.shape)   # (1024, 2, 256)  (1024, 256)  (1024,)
```

### Convert to Transformer input (integer token sequence)

```python
from latticeprobe.representations import to_sequence

tokens = to_sequence(a, b)
print(tokens.shape, tokens.dtype)   # torch.Size([768]) torch.int64
# Sequence order: k*n coefficients of a, then n coefficients of b
```

### Convert to GNN input (bipartite graph)

```python
from latticeprobe.representations import to_graph

graph = to_graph(a, b, params)
print(graph.x.shape)            # (768, 1)  — k*n variable + n equation nodes
print(graph.edge_index.shape)   # (2, 1024) — 2*k*n undirected edges
print(graph.edge_attr.shape)    # (1024, 1) — normalised coefficient weights
```

### Forward pass

```python
import torch
from latticeprobe.models.transformer import LWETransformer
from latticeprobe.models.gnn import LWEGNN

# Transformer
model_t = LWETransformer(params)
x = tokens.unsqueeze(0)                        # (1, 768)
logit = model_t(x)                             # (1, 1) raw binary logit

# GNN (requires torch_geometric DataLoader for batching)
model_g = LWEGNN(params)
from torch_geometric.data import Batch
batch = Batch.from_data_list([graph])
logit_g = model_g(batch)                       # (1, 1)
```

---

## Parameter sets

| Name | n | k | q | Noise | Notes |
|------|---|---|---|-------|-------|
| `ML-KEM-512` | 256 | 2 | 3329 | CBD η=3 | FIPS-203 standard |
| `ML-KEM-768` | 256 | 3 | 3329 | CBD η=2 | FIPS-203 standard |
| `ML-KEM-1024` | 256 | 4 | 3329 | CBD η=2 | FIPS-203 standard |
| `W1` | 256 | 2 | 3329 | CBD η=3 | Binary secret (Arora-Ge vulnerable) |
| `W2` | 256 | 2 | 3329 | Zero | No noise — trivially solvable |
| `W3` | 256 | 2 | 3329 | Gaussian σ=0.490 | σ reduced 60% from ML-KEM-512 |
| `edge` | 256 | 2 | 3329 | Gaussian σ=0.796 | σ reduced 35% (edge-of-margin) |

σ for ML-KEM-512 CBD(η=3) ≈ 1.225. W3 = 1.225 × 0.40 ≈ 0.490; edge = 1.225 × 0.65 ≈ 0.796.

---

## Architecture summary

### Transformer (`src/latticeprobe/models/transformer.py`)

| Hyperparameter | Value |
|---|---|
| Layers | 8 |
| Attention heads | 8 (paper spec: 12; changed because 512 % 12 ≠ 0) |
| Hidden dim | 512 |
| FFN dim | 2048 |
| Dropout | 0.1 |
| Input | Integer token sequence of length k·n + n |
| Head | Linear(512, 1) on CLS token |

### GNN (`src/latticeprobe/models/gnn.py`)

| Hyperparameter | Value |
|---|---|
| Backbone | GraphSAGE (Hamilton et al., 2017) |
| Layers | 6 |
| Hidden dim | 256 |
| Dropout | 0.1 |
| Input | Bipartite graph (k·n + n nodes, 2·k·n edges) |
| Readout | Global mean pool → Linear(256, 1) |

### Baselines (`src/latticeprobe/baselines.py`)

| Baseline | API |
|---|---|
| Logistic regression | `run_logistic_regression(A_train, B_train, y_train, A_test, B_test, y_test)` |
| MLP | `run_mlp(...)` |
| χ² distinguisher | `chi2_distinguisher(B_lwe, B_unif, params)` |
| Lattice estimator | `run_lattice_estimator(params)` — returns None if not installed |
| Bootstrap AUROC CI | `bootstrap_auroc(y_true, y_score, n_boot=100, ci=0.95)` → (mean, lo, hi) |

---

## Expected results (paper Table 2)

All AUROC values are means over 100 bootstrap resamples. 95% CIs omitted here; see paper.

| Parameter set | Transformer AUROC | GNN AUROC | Stat baseline |
|---|---|---|---|
| ML-KEM-512 | 0.502 | 0.501 | 0.500 |
| ML-KEM-768 | 0.500 | 0.500 | 0.500 |
| ML-KEM-1024 | 0.500 | 0.500 | 0.500 |
| Edge-of-margin (2¹⁸ samples) | — | 0.541 | 0.503 |
| W1 (binary secret) | ~0.71+ | ~0.71+ | — |
| W2 (no noise) | ~1.00 | ~1.00 | — |
| W3 (σ −60%) | ~0.85+ | ~0.85+ | — |

The standardised parameter results (≈ 0.50) confirm the hardness of ML-LWE at NIST-approved security levels.

---

## Project structure

```
latticeprobe/
├── src/latticeprobe/
│   ├── ring.py             # NTT, INTT, poly_mul, poly_add (FIPS-203 §4.3)
│   ├── params.py           # LWEParams dataclass + 7 parameter sets
│   ├── prng.py             # ChaCha20 CSPRNG wrapper (pynacl)
│   ├── sampler.py          # CBD, discrete_gaussian, generate_lwe_sample, generate_batch
│   ├── representations.py  # to_sequence(), to_graph()
│   ├── datasets.py         # LWESequenceDataset, LWEGraphDataset, save_shard()
│   ├── baselines.py        # LR, MLP, χ², lattice-estimator, bootstrap CI
│   └── models/
│       ├── transformer.py  # LWETransformer
│       └── gnn.py          # LWEGNN
├── tests/
│   ├── test_ring.py        # NTT round-trip, poly_mul vs naive (10+5+... = 20 tests)
│   ├── test_sampler.py     # CBD, Gaussian, LWE structure, batch (18 tests)
│   ├── test_representations.py  # sequence + graph (10 tests)
│   └── test_models.py      # transformer + GNN forward pass (11 tests)
├── docs/
│   ├── IMPL_NOTES.md       # Per-task implementation guidance derived from the paper
│   └── AI_Accelerated_...pdf
├── CHECKLIST.md
├── pyproject.toml
├── requirements.txt
└── compute_log.csv         # GPU-hour tracking (paper §5.3 requirement)
```

---

## Implementation notes

### NTT correctness

The negacyclic NTT follows FIPS-203 §4.3. One important discrepancy from the spec text:

- FIPS-203 Algorithm 42 (INTT) writes `−ζ` for the twiddle factor.
- The correct implementation (matching the pqcrystals/kyber C reference) uses **positive** `ζ`.

The sign difference arises from a different implicit convention in the spec vs. the C code. See `ring.py:83` for the comment and `tests/test_ring.py` for the round-trip tests that verify correctness.

### CSPRNG discipline

Every call to `generate_lwe_sample()` and `generate_uniform_sample()` calls `fresh_rng()`, which draws 32 bytes from pynacl's ChaCha20 source to seed a numpy `default_rng`. No RNG state is shared between samples.

### Disjoint key sets

`generate_batch()` generates a single secret `s` shared across all LWE samples in the batch. Callers are responsible for using **different secrets for train vs. test splits** (paper §5.2). Pass `secret=s_train` / `secret=s_test` explicitly.

---

## Reference

```bibtex
@article{ologunde2026latticeprobe,
  title   = {AI-Accelerated Cryptanalysis of Lattice-Based Schemes},
  author  = {Ologunde, E.},
  year    = {2026},
}
```
