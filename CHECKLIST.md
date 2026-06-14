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

- [ ] Implement polynomial ring arithmetic: `R_q = Z_q[X]/(X^n + 1)` with NTT multiplication
- [ ] Implement centred binomial distribution sampler (η parameter) for errors and secrets
- [ ] Implement `generate_lwe_sample(n, k, q, eta, secret=None)` → `(a, b)` in `R_q^k × R_q`
- [ ] Implement `generate_uniform_sample(n, k, q)` for the "random" class in the distinguisher
- [ ] **[Control]** Use ChaCha20-based CSPRNG: fresh seed per sample, no caching or deterministic shortcuts
- [ ] **[Control]** Cross-check output against the reference ML-KEM implementation on identical inputs
- [ ] **[Control]** Run χ² and KS tests on marginal + joint distributions of generated samples
- [ ] Implement three parameter classes:
  - Standardised: ML-KEM-512, ML-KEM-768, ML-KEM-1024 (exact FIPS-203 params)
  - Weakened (sanity): W1 (binary secret), W2 (no noise, σ=0), W3 (σ −60%)
  - Edge-of-margin: ML-KEM-512 with σ reduced 35%
- [ ] Produce dataset files at scales 2^16, 2^18, 2^20 per parameter class; store as sharded `.npz`
- [ ] **[Control]** Enforce disjoint key sets between train and test splits (different secret `s` per split)

### ML-KEM parameter reference

| Parameter set | n | k | q | η |
|---|---|---|---|---|
| ML-KEM-512 | 256 | 2 | 3329 | 3 |
| ML-KEM-768 | 256 | 3 | 3329 | 2 |
| ML-KEM-1024 | 256 | 4 | 3329 | 2 |

---

## Phase 3 — Data representations

- [ ] **Sequence repr (transformer):** flatten polynomial coefficients of `(a_i, b_i)` → integer sequence of length `k·n + n`
- [ ] Implement modular embedding: coefficients as integers in `[0, q)` with modular-arithmetic positional encoding (preferred over centred encoding per paper)
- [ ] **Graph repr (GNN):** construct bipartite graph — variable nodes (a coefficients) + equation nodes (b values), edges weighted by coefficient values mod q
- [ ] Wrap both representations as PyTorch `Dataset` / `DataLoader`; graph repr uses `torch_geometric.data.Data`

---

## Phase 4 — Model architectures

### ML models
- [ ] **Transformer:** 8-layer encoder, 12 attention heads, hidden dim 512, ~51M params — binary classifier head
- [ ] **GNN:** GraphSAGE backbone, 6 layers, hidden dim 256, ~18M params — binary classifier head

### Baselines (required for every result)
- [ ] Baseline 1: logistic regression on raw flattened coefficient features
- [ ] Baseline 2: MLP on flattened representations
- [ ] Baseline 3: χ² statistical distinguisher on coefficient distributions
- [ ] Baseline 4: lattice-estimator runtime prediction (call `lattice-estimator` CLI or Python API)

---

## Phase 5 — Training pipeline

- [ ] Loss: cross-entropy on the LWE-vs-uniform binary classification task
- [ ] Optimizer: AdamW with cosine learning-rate schedule
- [ ] Train for up to 50 epochs with validation-based early stopping (patience = 5)
- [ ] Log metrics to Weights & Biases (or TensorBoard): loss, AUROC, per-epoch
- [ ] Save checkpoints every 5 epochs; keep best-val-AUROC checkpoint for evaluation
- [ ] Document GPU-hours consumed per run in a `compute_log.csv`

---

## Phase 6 — Evaluation & controls

> **Run sanity checks first.** If the framework doesn't detect W1/W2/W3, you have a methodology bug, not a security finding.

- [ ] **[Sanity check]** Confirm W1 (binary secret), W2 (no noise), W3 (σ −60%) are detected at 2^16 and 2^18 — expected AUROC ≈ 0.71–1.0
- [ ] Evaluate all models on ML-KEM-512, ML-KEM-768, ML-KEM-1024 at 2^18 test samples
- [ ] Evaluate GNN on edge-of-margin regime at 2^14, 2^16, 2^18
- [ ] Compute **95% bootstrap confidence intervals** over 100 resamples for every AUROC reported
- [ ] **Cross-parameter generalisation:** train on each regime, evaluate on all others — fill 3×3 AUROC table
- [ ] **Partial-bit recovery:** train separate model to predict individual bits of secret vector; report per-bit accuracy vs 0.5 random baseline
- [ ] **Compute scaling experiment:** train on ML-KEM-512 at multiple GPU-hour budgets; plot AUROC vs compute (log scale)

### Expected results (from paper)

| Parameter | Transformer AUROC | GNN AUROC | Stat baseline |
|---|---|---|---|
| ML-KEM-512 | 0.502 | 0.501 | 0.500 |
| ML-KEM-768 | 0.500 | 0.500 | 0.500 |
| ML-KEM-1024 | 0.500 | 0.500 | 0.500 |
| Edge-of-margin (2^18) | — | 0.541 | 0.503 |

---

## Phase 7 — Open-source release

- [ ] Open-source the data-generation harness with full reproducibility instructions
- [ ] Open-source all baseline implementations (stat distinguisher, lattice-estimator wrapper, LR/MLP baselines)
- [ ] Release trained model weights (transformer + GNN) for all evaluated regimes via HuggingFace Hub or Zenodo
- [ ] Release evaluation harness so community can run stress-tests on new parameter sets
- [ ] Write a `REPRODUCE.md`: exact commands to regenerate every table and figure in the paper
- [ ] Pin all dependency versions; provide a Docker image for full env reproducibility

---

## Methodological obligations (per paper, section 8)

These are the minimum standards the paper sets for publishable ML-cryptanalysis work. Verify each before publishing results.

- [ ] Every ML result is compared to a classical statistical distinguisher baseline
- [ ] Framework is demonstrated to detect structure on classically-vulnerable parameter regimes
- [ ] Training and evaluation samples come from disjoint key sets (no memorisation)
- [ ] All AUROC values reported with bootstrap confidence intervals
- [ ] Training compute reported alongside results
- [ ] All code, data generation, trained models, and eval harness are open-source

---

*Based on: Ologunde, E. "AI-Accelerated Cryptanalysis of Lattice-Based Schemes." 2026.*