# LatticeProbe — Code Guide

A module-by-module walkthrough of every file in the project, how they connect, and the invariants you must not break.

---

## Project layout

```
src/latticeprobe/          ← importable library (install via pip install -e .)
  ring.py                  ← NTT / polynomial arithmetic
  params.py                ← LWEParams dataclass + 7 parameter sets
  prng.py                  ← ChaCha20 CSPRNG wrapper
  sampler.py               ← LWE and uniform sample generation
  representations.py       ← convert (a,b) → token tensor OR bipartite graph
  datasets.py              ← PyTorch Dataset wrappers for sharded .npz files
  baselines.py             ← LR, MLP, χ², lattice-estimator, bootstrap CI
  models/
    transformer.py         ← LWETransformer (paper §4.3)
    gnn.py                 ← LWEGNN / GraphSAGE (paper §4.3)

scripts/                   ← standalone CLI tools
  generate_dataset.py      ← write sharded .npz datasets to disk
  train.py                 ← training loop, checkpointing, compute log
  evaluate.py              ← load checkpoint → AUROC + baselines
  bit_recovery.py          ← per-bit logistic regression experiment

tests/
  test_ring.py             ← NTT round-trip, poly_mul algebraic laws
  test_sampler.py          ← CBD, Gaussian, LWE structure, batch shape
  test_representations.py  ← to_sequence / to_graph shapes and edge counts
  test_models.py           ← forward-pass shape, dtype, gradient, param count
  test_pipeline.py         ← end-to-end: generate → train → evaluate → recover
```

---

## Library modules

### `ring.py` — Polynomial arithmetic

**What it does:** implements R_q = Z_q[X]/(X^n+1) with n=256, q=3329.

```
Constants
  Q = 3329   N = 256   ZETA = 17
  ZETAS[k]  = 17^BitRev7(k) mod q       for k = 0..127  (NTT twiddle factors)
  GAMMAS[i] = 17^(2·BitRev7(i)+1) mod q for i = 0..127  (base-case quadratic)
  _INTT_FACTOR = 128^{-1} mod q = 3303

Public API
  ntt(f)              → fhat          # FIPS-203 Alg. 41
  intt(fhat)          → f             # FIPS-203 Alg. 42
  ntt_mul(fhat, ghat) → hhat          # pointwise in NTT domain (Alg. 11)
  poly_mul(f, g)      → h             # multiply via NTT: intt(ntt_mul(ntt(f),ntt(g)))
  poly_add(f, g)      → h             # (f+g) mod q, elementwise
  poly_mul_naive(f,g) → h             # O(n²) reference, used only in tests
```

**Critical invariant — INTT sign:** FIPS-203 Algorithm 42 writes `−ζ` as the twiddle factor. The *correct* implementation (matching pqcrystals C reference) uses **positive** `ZETAS[k]`. The code uses `ZETAS[k - n_groups + 1 : k + 1][::-1]` — do not negate these. The ten round-trip tests in `test_ring.py` verify this.

**Butterfly structure:** both NTT and INTT use the same vectorised reshape trick:
```
fhat.reshape(n_groups, 2, length)   → [top, bot] per group
```
This avoids an explicit loop over coefficients.

---

### `params.py` — Parameter sets

```python
@dataclass(frozen=True)
class LWEParams:
    name:   str
    n:      int           # polynomial degree (always 256)
    k:      int           # module rank (2/3/4)
    q:      int           # modulus (always 3329)
    noise:  "cbd" | "gaussian" | "zero"
    eta:    int = 0       # CBD width (used when noise="cbd")
    sigma:  float = 0.0   # Gaussian σ (used when noise="gaussian")
    secret: "cbd" | "binary" = "cbd"
```

Seven pre-defined sets accessed via `get_params(name)`:

| Name | k | noise | η / σ | secret | Purpose |
|---|---|---|---|---|---|
| ML-KEM-512 | 2 | cbd | 3 | cbd | FIPS-203 standard |
| ML-KEM-768 | 3 | cbd | 2 | cbd | FIPS-203 standard |
| ML-KEM-1024 | 4 | cbd | 2 | cbd | FIPS-203 standard |
| W1 | 2 | cbd | 3 | **binary** | Arora-Ge-vulnerable sanity check |
| W2 | 2 | **zero** | — | cbd | No noise — trivially solvable |
| W3 | 2 | **gaussian** | σ=0.490 | cbd | σ reduced 60% (η₅₁₂ ≈ 1.225 × 0.40) |
| edge | 2 | gaussian | σ=0.796 | cbd | σ reduced 35% (edge-of-margin) |

To add a new parameter set: add an entry to `PARAMS` and it becomes available in every script via `--param-set`.

---

### `prng.py` — CSPRNG

```python
fresh_rng() → np.random.Generator
```

Uses pynacl (ChaCha20) to draw 32 bytes, seeds numpy's `default_rng`. Called once per sample — **no shared RNG state**. This is a paper §5.1 control requirement. Never replace `fresh_rng()` with a module-level `np.random` call.

---

### `sampler.py` — Sample generation

**Three noise samplers:**

```python
cbd(eta, n, rng)                → int64 array ∈ [-η, η]
    # 2η bits → sum difference; exact FIPS-203 formula

discrete_gaussian(sigma, n, rng) → int64 array
    # round(Normal(0,σ²)), reject & resample tails beyond 6σ

_sample_noise(params, rng)      → int64 array   (internal)
_sample_secret(params, rng)     → int64 array   (internal)
```

**Sample generators (public):**

```python
generate_lwe_sample(params, secret=None)
    → (a, b, s)
    # a: (k, n) uniform in [0, q)
    # b: (n,)  = e + Σ_r poly_mul(a[r], s[r])  mod q
    # s: (k, n) — returned so callers can fix it across a split

generate_uniform_sample(params)
    → (a, b)
    # a: (k, n) uniform;  b: (n,) uniform — no algebraic relation

generate_batch(params, n_samples, secret=None)
    → (A, B, labels)
    # A: float32 (N, k, n)  B: float32 (N, n)  labels: int8 (N,) ∈ {0,1}
    # Uses ONE fixed secret for all LWE rows.
    # If secret=None, draws a fresh one and returns it embedded in the first call.
```

**Disjoint key invariant:** train, val, and test splits must call `generate_batch` (or `generate_dataset`) independently — each gets a fresh secret. Never pass the train secret to the test call.

---

### `representations.py` — Input converters

Two functions that convert a single `(a, b)` sample into model input:

**Sequence (for Transformer):**
```python
to_sequence(a, b) → torch.int64 tensor, shape (k*n + n,)
    # Flat concatenation: a.reshape(-1) ++ b
    # Tokens are raw integer coefficients in [0, q) — modular embedding
```

**Bipartite graph (for GNN):**
```python
to_graph(a, b, params) → torch_geometric.data.Data
    # Nodes:
    #   [0 .. k*n-1]       variable nodes, feature = a[r,c] / q
    #   [k*n .. k*n+n-1]   equation nodes, feature = b[c] / q
    # Edges: variable (r*n+c) <-> equation (k*n+c), undirected
    # Edge attr = a[r,c] / q (same value both directions)
    # Total edges = 2*k*n  (both directions)
```

For ML-KEM-512 (k=2, n=256):
- Sequence length: 768
- Graph nodes: 768, edges: 1024 (× 2 undirected = 2048 entries in edge_index)

---

### `datasets.py` — Dataset wrappers

Both classes load all shards into RAM at construction time. Suitable for scale ≤ 2^18 (~1.5 GB RAM for k=2).

```python
class LWESequenceDataset(Dataset):
    # Returns: (tokens: int64 tensor (seq_len,), label: float tensor scalar)
    # Use with: torch.utils.data.DataLoader

class LWEGraphDataset(Dataset):
    # Returns: (graph: Data, label: float tensor (1,))
    # Use with: torch_geometric.loader.DataLoader  ← MUST use this, not standard

def save_shard(path, a, b, labels):
    # Writes compressed .npz: a=float32, b=float32, label=int8
```

**Shard format** (every `.npz` file):
```
a:     float32  (shard_size, k, n)
b:     float32  (shard_size, n)
label: int8     (shard_size,)   0=uniform, 1=LWE
```

---

### `baselines.py` — Baselines and utilities

```python
# Classical baselines
run_logistic_regression(A_train, B_train, y_train, A_test, B_test, y_test) → dict
run_mlp(...)                                                                 → dict
chi2_distinguisher(B_lwe, B_unif, params)                                    → dict
run_lattice_estimator(params)     → dict | None   # None if not installed

# Evaluation utility
bootstrap_auroc(y_true, y_score, n_boot=100, ci=0.95) → (mean, lo, hi)
    # 100 resamples, percentile bootstrap (not BCa)
    # seed=0 → reproducible CI

# Feature helper (used internally)
flatten(A, B) → float32 array (N, k*n + n)
```

`chi2_distinguisher` uses per-sample mean absolute deviation from q/2 as a distinguishing score — this is the AUROC proxy, not the p-value.

`run_lattice_estimator` requires the `lattice-estimator` package (not in requirements.txt). Returns None if absent, so it is always safe to call.

---

### `models/transformer.py` — LWETransformer

```
Architecture
  nn.Embedding(q, d_model)         ← token embedding, vocab = q = 3329
  nn.Embedding(seq_len, d_model)    ← learned positional embedding
  nn.TransformerEncoder
    num_layers=8
    d_model=512
    nhead=8  (paper says 12 but 512 % 12 ≠ 0 — using 8)
    dim_feedforward=2048
    batch_first=True, norm_first=True  (Pre-LN)
  nn.LayerNorm(d_model)
  nn.Linear(d_model, 1)             ← classification head on position 0 (CLS)

forward(x: int64 (B, seq_len)) → float32 (B, 1)   # raw logit
```

CLS position: position 0 (the first token) acts as the classification token — `h[:, 0]` is fed to the head, consistent with BERT-style pre-LN transformers.

Weight init: token/pos embeddings use N(0, 0.02); head bias zero.

Constructor args are all stored as defaults in the signature — `train.py` reads them from `args` and saves them in every checkpoint so `evaluate.py` can reconstruct the exact same model size.

---

### `models/gnn.py` — LWEGNN

```
Architecture (GraphSAGE)
  SAGEConv(1, hidden)          ← layer 0
  SAGEConv(hidden, hidden) × 5 ← layers 1–5
  BatchNorm1d(hidden) per layer
  Dropout(p=0.1) per layer
  global_mean_pool              ← aggregate all node embeddings
  nn.Linear(hidden, 1)         ← classification head

forward(data: Batch) → float32 (B, 1)   # raw logit
    # data.x:          node features (total_nodes, 1)
    # data.edge_index: (2, total_edges)
    # data.batch:      node-to-graph assignment
```

Defaults: hidden=256, num_layers=6, dropout=0.1.

Note: `edge_attr` (normalised a[r,c]/q) is present in the Data objects but **not used** by SAGEConv — SAGEConv aggregates node features only. Edge attributes are available for future experiments with edge-feature convolutions.

---

## Script modules

### `scripts/generate_dataset.py`

```
generate_dataset(param_set, n_samples, output_dir, shard_size=4096, quiet=False)
    → Path (output directory)
```

Flow:
1. Draw one fixed secret `s` via `generate_lwe_sample(params)`.
2. Save `secret.npy` to the output directory (needed by `bit_recovery.py`).
3. Loop in chunks of `shard_size`, call `generate_batch(params, batch, secret=s)`.
4. Write each chunk as `shard_NNNNN.npz` via `save_shard()`.

CLI: `python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 65536 --output-dir data/train`

---

### `scripts/train.py`

```
train(args) → Path   # path to best.pt
```

Key functions:
```python
build_model(model_name, params, device, args) → nn.Module
    # reads d_model/nhead/num_layers/ff_dim from args for transformer
    # reads hidden/gnn_layers from args for gnn

build_loader(model_name, data_dir, params, batch_size, shuffle) → DataLoader
    # uses torch_geometric.loader.DataLoader for GNN

run_epoch(model, loader, device, model_name, optimizer=None) → (loss, auroc)
    # optimizer=None → eval mode (no_grad)
    # gradient clipping: max_norm=1.0
```

Training loop:
- AdamW(lr=1e-4, wd=1e-2) + CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
- Early stopping on val AUROC with configurable patience
- Every N epochs: save `ckpt_epochNNN.pt`
- On improvement: overwrite `best.pt`
- Each epoch: append one row to `compute_log.csv`

**Checkpoint format** — must contain these keys:
```python
{
    "epoch":          int,
    "val_auroc":      float,
    "model_state":    dict,    # model.state_dict()
    "args":           dict,    # vars(args) — needed by evaluate.py
}
```

The `"args"` key is critical: `evaluate.py` reads `d_model`, `nhead`, `num_layers`, `ff_dim`, `hidden`, `gnn_layers` from it to reconstruct the exact model size. Periodic checkpoints also save `"optimizer_state"` for resuming.

---

### `scripts/evaluate.py`

```
evaluate_model(checkpoint, model_name, param_set, test_dir,
               train_dir=None, batch_size=256, n_boot=100, device_arg="auto")
    → dict
```

Return dict keys:
```
param_set, model, checkpoint, n_test
model_auroc, model_auroc_ci_lo, model_auroc_ci_hi     ← always present
chi2_auroc, chi2_ci_lo, chi2_ci_hi                    ← always present
lr_auroc, lr_ci_lo, lr_ci_hi                          ← if train_dir provided
mlp_auroc, mlp_ci_lo, mlp_ci_hi                       ← if train_dir provided
```

The χ² distinguishing score is mean absolute deviation of b coefficients from q/2.

`load_checkpoint(path, model_name, params, device)` reads `ckpt["args"]` to get the architecture dimensions used during training — this is how you avoid size mismatch errors when the test model uses different dims than the default.

---

### `scripts/bit_recovery.py`

```
recover_bits(train_dir, test_dir, param_set, bit_range=None) → dict
```

For each bit position `i` in `[0, k*n)`:
1. Target label = `(secret_flat[i] > 0)` — sign of the secret coefficient.
2. Features = `[a.reshape(-1) / q, b / q]` (from LWE samples only, labels==1).
3. Fit `LogisticRegression` on train, measure accuracy on test.

`bit_range` is a Python slice string, e.g. `"0:32"` — tests only the first 32 bits.

Expected result on standardised params: mean accuracy ≈ 0.500 (no leakage).

---

## Data flow

```
generate_dataset.py
  └─ generate_batch()
       ├─ generate_lwe_sample()   (label=1)
       │     └─ poly_mul() via NTT
       └─ generate_uniform_sample() (label=0)
  └─ save_shard() → shard_*.npz + secret.npy

train.py
  ├─ LWESequenceDataset OR LWEGraphDataset  (reads shards)
  │     └─ to_sequence() OR to_graph()     (per __getitem__)
  ├─ build_model() → LWETransformer OR LWEGNN
  ├─ run_epoch() → loss, AUROC
  └─ saves best.pt with "args" key

evaluate.py
  ├─ load_checkpoint() → model (reconstructed from saved args)
  ├─ _inference() → logits, labels
  ├─ bootstrap_auroc() → mean, lo, hi
  └─ chi2_distinguisher() + LR/MLP baselines

bit_recovery.py
  ├─ _load_arrays() → A, B (LWE samples only)
  ├─ _load_secret() → secret.npy
  └─ per-bit LogisticRegression
```

---

## Test structure

| File | Tests | What's covered |
|---|---|---|
| test_ring.py | 20 | NTT round-trip ×10, poly_mul vs naive ×5, algebraic laws |
| test_sampler.py | 18 | CBD range & distribution (χ²), Gaussian, LWE structure, W2, batch |
| test_representations.py | 10 | to_sequence shape, to_graph node/edge counts |
| test_models.py | 11 | Transformer + GNN shape, dtype, no-NaN, param count, gradients |
| test_pipeline.py | 19 | end-to-end generate/train/evaluate/recover on tiny CPU models |

**Pipeline test architecture:** tests use a stripped-down model (`d_model=64, nhead=4, num_layers=2, ff_dim=128, hidden=32, gnn_layers=2`) so they run in under 90 s on CPU. This is controlled by `_make_args()` in `test_pipeline.py`.

Run all tests:
```bash
pytest tests/ -q
# 81 passed expected
```

---

## Key invariants

**1. Fresh RNG per sample.** `fresh_rng()` is called once at the start of `generate_lwe_sample()` and `generate_uniform_sample()`. Never move this call outside the per-sample scope.

**2. Fixed secret per split.** `generate_dataset.py` draws one secret, saves `secret.npy`, and passes it to every `generate_batch()` call within that split. Train, val, and test must be in separate output directories — each gets an independent secret.

**3. INTT twiddle sign.** Use positive `ZETAS` in INTT (line 83 of ring.py). Negating them gives wrong results that are not caught by casual inspection.

**4. Checkpoint stores `"args"`.** `evaluate.py` reads architecture dimensions from `ckpt["args"]`. If you add a new architecture parameter, add it to both `train.py`'s `parse_args()` and `build_model()`, so it flows into the checkpoint and is recovered at eval time.

**5. GNN needs torch_geometric DataLoader.** Using `torch.utils.data.DataLoader` with `LWEGraphDataset` breaks batching — `Data` objects must be batched via `torch_geometric.loader.DataLoader`.

**6. Labels are float, shaped (B, 1).** `BCEWithLogitsLoss` requires matching shapes. Both model types enforce this in `run_epoch()`: transformer does `.unsqueeze(1)`, GNN does `.reshape(-1, 1)`.

---

## How to extend

**New parameter set:** add one `LWEParams` entry to `PARAMS` in `params.py`. Everything else picks it up automatically.

**New noise type:** add a branch in `_sample_noise()` in `sampler.py` and update the `noise` literal type in `LWEParams`.

**New model:** implement `forward(x) → (B, 1)` logit tensor. Add a branch in `build_model()` in `train.py` and `load_checkpoint()` in `evaluate.py`. Register new CLI args in both scripts' `parse_args()`.

**Larger dataset scale:** shards are independent, so generation is embarrassingly parallel. The dataset classes concatenate everything into RAM — for scales > 2^20 consider lazy shard loading (memmap or an `IterableDataset`).
