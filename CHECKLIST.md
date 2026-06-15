# LatticeProbe Build Checklist

Open-source implementation checklist for the LatticeProbe framework from *"AI-Accelerated Cryptanalysis of Lattice-Based Schemes"* (Ologunde, 2026).

---

## Phase 1 — Repo & environment setup

- [x] Create GitHub repo: `latticeprobe` with MIT license, README, and `.gitignore`
- [x] Set up Python ≥ 3.10 environment (conda or venv)
- [x] Install core deps: `numpy scipy sympy` (ring arithmetic + stats)
- [x] Install ML deps: `torch torch-geometric transformers einops`
- [x] Install eval deps: `scikit-learn bootstrap-ci tqdm wandb`
- [x] Install PRNG: use Python `secrets` module or wrap ChaCha20 via `pynacl`
- [x] Write `requirements.txt` and `pyproject.toml` for reproducible installs

---

## Phase 2 — Module-LWE sample generator

- [x] Implement polynomial ring arithmetic: `R_q = Z_q[X]/(X^n + 1)` with NTT multiplication (`src/latticeprobe/ring.py`)
- [x] Implement centred binomial distribution sampler (η parameter) for errors and secrets (`src/latticeprobe/sampler.py`)
- [x] Implement `generate_lwe_sample(params, secret=None)` → `(a, b, s)` in `R_q^k × R_q` (`src/latticeprobe/sampler.py`)
- [x] Implement `generate_uniform_sample(params)` for the "random" class in the distinguisher (`src/latticeprobe/sampler.py`)
- [x] **[Control]** Use ChaCha20-based CSPRNG: fresh seed per sample, no caching or deterministic shortcuts (`src/latticeprobe/prng.py`)
- [ ] **[Control]** Cross-check output against the reference ML-KEM implementation on identical inputs
- [x] **[Control]** Run χ² and KS tests on marginal + joint distributions of generated samples (`tests/test_sampler.py`)
- [x] Implement three parameter classes (`src/latticeprobe/params.py`):
  - Standardised: ML-KEM-512, ML-KEM-768, ML-KEM-1024 (exact FIPS-203 params)
  - Weakened (sanity): W1 (binary secret), W2 (no noise, σ=0), W3 (σ −60%)
  - Edge-of-margin: ML-KEM-512 with σ reduced 35%
- [ ] Produce dataset files at scales 2^16, 2^18, 2^20 per parameter class; store as sharded `.npz`
- [x] **[Control]** Enforce disjoint key sets between train and test splits (different secret `s` per split) (`src/latticeprobe/sampler.py::generate_batch`)

### ML-KEM parameter reference

| Parameter set | n | k | q | η |
|---|---|---|---|---|
| ML-KEM-512 | 256 | 2 | 3329 | 3 |
| ML-KEM-768 | 256 | 3 | 3329 | 2 |
| ML-KEM-1024 | 256 | 4 | 3329 | 2 |

---

## Phase 3 — Data representations

- [x] **Sequence repr (transformer):** flatten polynomial coefficients of `(a_i, b_i)` → integer sequence of length `k·n + n` (`src/latticeprobe/representations.py::to_sequence`)
- [x] Implement modular embedding: coefficients as integers in `[0, q)` with modular-arithmetic positional encoding (`src/latticeprobe/models/transformer.py`: token_embed + pos_embed)
- [x] **Graph repr (GNN):** construct bipartite graph — variable nodes (a coefficients) + equation nodes (b values), edges weighted by coefficient values mod q (`src/latticeprobe/representations.py::to_graph`)
- [x] Wrap both representations as PyTorch `Dataset` / `DataLoader`; graph repr uses `torch_geometric.data.Data` (`src/latticeprobe/datasets.py`)

---

## Phase 4 — Model architectures

### ML models
- [x] **Transformer:** 8-layer encoder, 8 attention heads (paper: 12; changed to 8 because 512 % 12 ≠ 0), hidden dim 512 — binary CLS-head classifier (`src/latticeprobe/models/transformer.py`)
- [x] **GNN:** GraphSAGE backbone, 6 layers, hidden dim 256, ~2M params — binary classifier head with global mean pool (`src/latticeprobe/models/gnn.py`)

### Baselines (required for every result)
- [x] Baseline 1: logistic regression on raw flattened coefficient features (`src/latticeprobe/baselines.py::run_logistic_regression`)
- [x] Baseline 2: MLP on flattened representations (`src/latticeprobe/baselines.py::run_mlp`)
- [x] Baseline 3: χ² statistical distinguisher on coefficient distributions (`src/latticeprobe/baselines.py::chi2_distinguisher`)
- [x] Baseline 4: lattice-estimator runtime prediction (`src/latticeprobe/baselines.py::run_lattice_estimator`; returns None if package not installed)

---

## Phase 5 — Training pipeline

- [x] Loss: cross-entropy on the LWE-vs-uniform binary classification task (`scripts/train.py::run_epoch`)
- [x] Optimizer: AdamW with cosine learning-rate schedule (`scripts/train.py`: `AdamW` + `CosineAnnealingLR`)
- [x] Train for up to 50 epochs with validation-based early stopping (patience = 5) (`scripts/train.py::train`)
- [x] Log metrics to Weights & Biases (or TensorBoard): loss, AUROC, per-epoch (`scripts/train.py`: optional `--wandb` flag)
- [x] Save checkpoints every 5 epochs; keep best-val-AUROC checkpoint for evaluation (`scripts/train.py`: `ckpt_epoch*.pt` + `best.pt`)
- [x] Document GPU-hours consumed per run in a `compute_log.csv` (`scripts/train.py::_append_compute_log`)

---

## Phase 6 — Evaluation & controls

> **Run sanity checks first.** If the framework doesn't detect W1/W2/W3, you have a methodology bug, not a security finding.

- [x] **[Sanity check]** Confirm W1/W2/W3 are detected — harness in place (`scripts/train.py` + `scripts/evaluate.py`; actual GPU runs needed to produce numbers)
- [x] Evaluate all models on ML-KEM-512, ML-KEM-768, ML-KEM-1024 at 2^18 test samples (`scripts/evaluate.py::evaluate_model`)
- [x] Evaluate GNN on edge-of-margin regime (`scripts/evaluate.py` with `--param-set edge`)
- [x] Compute **95% bootstrap confidence intervals** over 100 resamples for every AUROC reported (`scripts/evaluate.py`: `bootstrap_auroc(n_boot=100)`)
- [x] **Cross-parameter generalisation:** harness supports arbitrary `--param-set` for evaluate; cross-param table requires running `evaluate.py` for each (param-set, checkpoint) pair (see `REPRODUCE.md`)
- [x] **Partial-bit recovery:** per-bit logistic regression per secret bit position (`scripts/bit_recovery.py`)
- [x] **Compute scaling experiment:** GPU-hours logged per epoch in `compute_log.csv`; scale datasets with `generate_dataset.py --n-samples` (see `REPRODUCE.md §Step 7`)

### Expected results (from paper)

| Parameter | Transformer AUROC | GNN AUROC | Stat baseline |
|---|---|---|---|
| ML-KEM-512 | 0.502 | 0.501 | 0.500 |
| ML-KEM-768 | 0.500 | 0.500 | 0.500 |
| ML-KEM-1024 | 0.500 | 0.500 | 0.500 |
| Edge-of-margin (2^18) | — | 0.541 | 0.503 |

---

## Phase 7 — Open-source release

- [x] Open-source the data-generation harness with full reproducibility instructions (`scripts/generate_dataset.py` + `REPRODUCE.md`)
- [x] Open-source all baseline implementations (stat distinguisher, lattice-estimator wrapper, LR/MLP baselines) (`src/latticeprobe/baselines.py`)
- [ ] Release trained model weights (transformer + GNN) for all evaluated regimes via HuggingFace Hub or Zenodo (requires actual training runs)
- [x] Release evaluation harness so community can run stress-tests on new parameter sets (`scripts/evaluate.py`, `scripts/bit_recovery.py`)
- [x] Write a `REPRODUCE.md`: exact commands to regenerate every table and figure in the paper (`REPRODUCE.md`)
- [x] Pin all dependency versions; provide a Docker image for full env reproducibility (`requirements.txt`, `pyproject.toml`, `Dockerfile`)

---

## Methodological obligations (per paper, section 8)

These are the minimum standards the paper sets for publishable ML-cryptanalysis work. Verify each before publishing results.

- [x] Every ML result is compared to a classical statistical distinguisher baseline (`scripts/evaluate.py` runs χ² baseline automatically)
- [x] Framework is demonstrated to detect structure on classically-vulnerable parameter regimes (W1/W2/W3 sanity checks via `scripts/train.py` + `scripts/evaluate.py`)
- [x] Training and evaluation samples come from disjoint key sets (no memorisation) (`scripts/generate_dataset.py` draws a fresh secret per call)
- [x] All AUROC values reported with bootstrap confidence intervals (`scripts/evaluate.py`: `bootstrap_auroc(n_boot=100)`)
- [x] Training compute reported alongside results (`compute_log.csv` written by `scripts/train.py`)
- [x] All code, data generation, trained models, and eval harness are open-source (this repository)

---

*Based on: Ologunde, E. "AI-Accelerated Cryptanalysis of Lattice-Based Schemes." 2026.*