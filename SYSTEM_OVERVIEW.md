# LatticeProbe — System Overview

Source of truth: *"AI-Accelerated Cryptanalysis of Lattice-Based Schemes: A Stress-Test of NIST PQC Parameter Choices via Transformer and Graph Neural Network Distinguishers"* (Ologunde, 2026), verified from PDF.

---

## 1. What LatticeProbe Is

A research framework for **ML-based cryptanalysis stress-testing** of NIST FIPS-203 (ML-KEM) parameters. The distinguishing task: given a pair `(a, b)`, classify it as:

- **Label 1** — a real LWE sample: `b = <a, s> + e mod q`
- **Label 0** — a uniform random pair: `b` drawn uniformly from `R_q`

The expected result at standardised ML-KEM parameters: **AUROC ≈ 0.500** (no detectable structure). The framework verifies this null result rigorously, with calibrated baselines and sanity checks on deliberately weakened regimes.

**Paper headline (Table 2):**

| Parameter | Transformer AUROC | GNN AUROC | Stat baseline |
|-----------|------------------|-----------|---------------|
| ML-KEM-512 | 0.502 | 0.501 | 0.500 |
| ML-KEM-768 | 0.500 | 0.500 | 0.500 |
| ML-KEM-1024 | 0.500 | 0.500 | 0.500 |

**Edge-of-margin signal (Table 3, σ reduced 35%):** GNN AUROC 0.541 [0.532, 0.549] at 2^18 samples — small but statistically significant; vanishes at standardised parameters.

---

## 2. Paper vs Implementation — Full Audit Table

| Component | Paper (§) | Implementation | Status |
|-----------|-----------|----------------|--------|
| Transformer layers | 8 (§4.3) | 8 | ✅ MATCH |
| Transformer d\_model | 512 (§4.3) | 512 | ✅ MATCH |
| Transformer nhead | **12** (§4.3) | **8** | ⚠️ PAPER INCONSISTENCY — 512%12≠0; 8 is the only valid value |
| Transformer ff\_dim | NOT SPECIFIED | 2048 | — NOT IN PAPER |
| Transformer parameters | **~51M** (§4.3) | **~27M** | ⚠️ PAPER INCONSISTENCY |
| Transformer CLS token | NOT SPECIFIED | Learnable CLS prepended | 🔧 ADDED (architecturally principled) |
| GNN backbone | **GraphSAGE** (§4.3, §9.2) | SAGEConv | ✅ FIXED (was GATConv) |
| GNN layers | 6 (§4.3) | 6 | ✅ MATCH |
| GNN hidden dim | 256 (§4.3) | 256 | ✅ MATCH |
| GNN parameters | **~18M** (§4.3) | **~0.66M** | ⚠️ PAPER INCONSISTENCY |
| Sequence encoding | Modular [0,q) (§4.2.1) | [0,q) tokens | ✅ MATCH |
| Secret per split | Disjoint key sets (§5.2) | Fresh secret per split | ✅ FIXED (was shared) |
| W1 regime | Binary secret, σ = ML-KEM-512 (§5.4) | Binary, CBD(η=3) noise | ✅ MATCH |
| W2 regime | σ = 0, no noise (§5.4) | σ = 0 | ✅ MATCH |
| W3 regime | σ reduced **60%** (§5.4) | σ = 0.490 = 1.2247×0.40 | ✅ MATCH |
| Edge-of-margin | σ reduced **35%** (§4.1) | σ = 0.796 = 1.2247×0.65 | ✅ MATCH |
| Optimizer | AdamW + cosine LR (§4.4) | AdamW + CosineAnnealingLR | ✅ MATCH |
| Max epochs | ~50 (§4.4) | 50 (default) | ✅ MATCH |
| Early stopping | Validation-based (§4.4) | patience=5 | ✅ MATCH |
| Training samples | 2^20 (§4.4) | Configurable (demo: 2^14) | ℹ️ SCALE ONLY |
| Test samples | 2^18 (§3.2) | Configurable (demo: 8192) | ℹ️ SCALE ONLY |
| Bootstrap CI | 95%, 100 resamples (§5.2) | BCa bootstrap, 100 resamples | ✅ MATCH |
| Baselines | LR, MLP, χ², lattice estimator (§4.3) | All implemented | ✅ MATCH |
| Dataset dtype | NOT SPECIFIED | int16 a/b, int8 labels | — |

---

## 3. Critical Mismatches & Fixes

### Fix 1 — GNN backbone: GATConv → GraphSAGE (SAGEConv)

**What the paper says (§4.3):** "GNN: GraphSAGE backbone with 6 layers, hidden dim 256."
Confirmed again in §9.2: *"The GNN representation we evaluate is a generic GraphSAGE."*

**What the old code had:**
```python
# WRONG — old src/latticeprobe/models/gnn.py
from torch_geometric.nn import GATConv, global_mean_pool

class LWEGNN(nn.Module):
    def __init__(self, params, hidden=256, num_layers=6, dropout=0.1):
        ...
        self.convs.append(GATConv(in_ch, hidden, edge_dim=1))  # GAT, not SAGE

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)  # GAT needs edge features
```

**Why this matters:** GATConv (Graph Attention Networks) and GraphSAGE are fundamentally different algorithms:
- **GAT** computes per-edge attention scores; requires `edge_dim`; learns which neighbours to attend to
- **GraphSAGE** aggregates neighbours by mean; ignores edge features; inductive generalisation

Training with GATConv produces results that are **not reproducible** against the paper.

**Fixed code — `src/latticeprobe/models/gnn.py`:**
```python
from torch_geometric.nn import SAGEConv, global_mean_pool  # ← SAGEConv, not GATConv

class LWEGNN(nn.Module):
    def __init__(self, params, hidden=256, num_layers=6, dropout=0.1):
        super().__init__()
        in_channels = 1
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden
            self.convs.append(SAGEConv(in_ch, hidden))   # ← no edge_dim
            self.norms.append(nn.BatchNorm1d(hidden))
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)   # ← SAGEConv takes no edge_attr
            x = norm(x)
            x = x.relu()
            x = self.drop(x)
        x = global_mean_pool(x, batch)
        return self.head(x)
```

**Parameter count:** ~0.66M (paper claims 18M — see §4 for analysis).

---

### Fix 2 — Secret sharing across splits violates paper §5.2

**What the paper says (§5.2):** *"Disjoint key sets: training and test samples are produced under different secret keys to ensure cross-key generalisation rather than memorisation."* (Repeated in §3.2.)

**What the old notebook had (cell-13):**
```python
# WRONG — old notebook/Lattice_Probe_Experiments.ipynb cell-13

# Val used train's secret
subprocess.run([
    "python", "scripts/generate_dataset.py",
    "--param-set", PARAM_SET,
    "--n-samples", str(VAL_SAMPLES),
    "--output-dir", val_dir,
    "--secret-file", f"{train_dir}/secrets.npy"   # ← shares train secret
], check=True)

# Test also used train's secret
subprocess.run([
    "python", "scripts/generate_dataset.py",
    "--param-set", PARAM_SET,
    "--n-samples", str(TEST_SAMPLES),
    "--output-dir", test_dir,
    "--secret-file", f"{train_dir}/secrets.npy"   # ← shares train secret
], check=True)
```

**Why this matters:** When train, val, and test all share the same secret `s`, the model may be memorising `s` rather than learning general LWE structure. The cross-key generalisation test (Section 7) becomes circular — the "cross-secret" comparison has no meaning if they were never actually cross-key.

**Fixed code — notebook cell-13:**
```python
# CORRECT — each split draws its own fresh independent secret

if not os.path.exists(f"{train_dir}/secrets.npy"):
    subprocess.run([
        "python", "scripts/generate_dataset.py",
        "--param-set", PARAM_SET,
        "--n-samples", str(TRAIN_SAMPLES),
        "--output-dir", train_dir,
        # no --secret-file: fresh secret drawn here
    ], check=True)

# Fresh secret per split — paper §5.2 requires disjoint key sets
if not os.path.exists(f"{val_dir}/secrets.npy"):
    subprocess.run([
        "python", "scripts/generate_dataset.py",
        "--param-set", PARAM_SET,
        "--n-samples", str(VAL_SAMPLES),
        "--output-dir", val_dir,
        # no --secret-file: independent secret for val
    ], check=True)

if not os.path.exists(f"{test_dir}/secrets.npy"):
    subprocess.run([
        "python", "scripts/generate_dataset.py",
        "--param-set", PARAM_SET,
        "--n-samples", str(TEST_SAMPLES),
        "--output-dir", test_dir,
        # no --secret-file: independent secret for test
    ], check=True)
```

---

### Fix 3 — Transformer training OOM on T4

**What happens:** With default `--batch-size 256`, training crashes immediately with:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB.
GPU 0 has 14.56 GiB total capacity; 819 MiB free.
```
(confirmed in stored notebook output, cell-15 return code 1)

**Why it happens:** With seq\_len = 769 (768 content + 1 CLS), d\_model=512, 8 layers:
- FFN intermediate activations per layer: (256, 769, 2048) × fp16 ≈ **806 MB per layer**
- 8 layers forward + backward: ~806 × 8 × 2 = **12.9 GB** for FFN alone
- Plus attention, norms, optimizer states: exceeds 15 GB

**Fix — notebook cell-15:**
```python
# WRONG — old default silently uses batch_size=256 → OOM on T4
subprocess.run([
    "python", "scripts/train.py",
    "--param-set", PARAM_SET,
    "--model", MODEL,
    "--train-dir", train_dir,
    "--val-dir", val_dir,
    "--output-dir", ckpt_out,
    "--epochs", "20",
    "--compute-log", f"{ckpt_out}/compute_log.csv",
], ...)

# CORRECT — explicit batch_size=32 fits on T4 (15 GB)
# At batch=32: FFN activations ≈ 100 MB/layer × 8 = 800 MB total → feasible
subprocess.run([
    "python", "scripts/train.py",
    "--param-set", PARAM_SET,
    "--model", MODEL,
    "--train-dir", train_dir,
    "--val-dir", val_dir,
    "--output-dir", ckpt_out,
    "--epochs", "20",
    "--batch-size", "32",      # ← added: 256 OOMs on T4 at seq_len=769
    "--compute-log", f"{ckpt_out}/compute_log.csv",
], ...)
```

---

### Fix 4 — Lazy LWEGNN import in train.py and evaluate.py

**What happened:** On Google Colab (PyTorch 2.x + cu128), `torch_geometric.nn` requires `torch_sparse` compiled for the exact CUDA version. This extension often fails to install. The old code imported `LWEGNN` unconditionally at module level, so **every** call to `train.py` or `evaluate.py` — even transformer-only runs — would crash on import with `ModuleNotFoundError` before printing a single line.

**Old train.py `__main__` block:**
```python
# WRONG — old scripts/train.py
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from latticeprobe.datasets import LWEGraphDataset, LWESequenceDataset
    from latticeprobe.models.transformer import LWETransformer
    from latticeprobe.models.gnn import LWEGNN   # ← always imported, crashes if torch_sparse missing
```

**Fixed train.py — lazy import inside `build_model()`:**
```python
# CORRECT — scripts/train.py

def build_model(model_name, params, device, args=None):
    if model_name == "transformer":
        return LWETransformer(params, **kwargs).to(device)
    # Lazy import: only reaches here when training a GNN.
    # Deferred to avoid requiring torch_sparse when training the transformer.
    from latticeprobe.models.gnn import LWEGNN   # ← only imported for GNN runs
    return LWEGNN(params, **kwargs).to(device)

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from latticeprobe.datasets import LWEGraphDataset, LWESequenceDataset
    from latticeprobe.models.transformer import LWETransformer
    # LWEGNN imported lazily inside build_model() — not here
```

**Fixed evaluate.py — lazy import inside `load_checkpoint()`:**
```python
# CORRECT — scripts/evaluate.py

def load_checkpoint(path, model_name, params, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    if model_name == "transformer":
        model = LWETransformer(params, ...)   # no LWEGNN needed
    else:
        from latticeprobe.models.gnn import LWEGNN   # ← only imported for GNN
        model = LWEGNN(params, ...)
    model.load_state_dict(ckpt["model_state"])
    return model, ...
```

---

## 4. Moderate Mismatches

### Mismatch A — Transformer CLS token not specified in paper

**What the paper says (§4.3):** "8-layer encoder, 12 attention heads, hidden dim 512. Trained as a binary classifier with cross-entropy loss." — No mention of CLS token, mean pooling, or any specific readout.

**What the old code had:**
```python
# AMBIGUOUS — old transformer.py forward()
# Position 0 was the first content token (a[0,0] coefficient), not a CLS token
h = self.encoder(h)     # h shape: (B, k*n+n, d_model)
h = self.norm(h[:, 0])  # reads position 0 — but position 0 = first coefficient of a
return self.head(h)
```

**Why this is wrong:** Using the first content position as a classification aggregator is architecturally inconsistent. Position 0 has a specific semantic meaning (first coefficient of `a[0,0]`). The encoder has no reason to aggregate global information there — nothing in the attention structure signals that it should.

**Fixed code — `src/latticeprobe/models/transformer.py`:**
```python
class LWETransformer(nn.Module):
    def __init__(self, params, d_model=512, nhead=8, num_layers=8,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        ...
        # Learnable CLS token prepended to every sequence before encoding.
        # Shape (1, 1, d_model) — broadcast over batch dimension.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))   # ← ADDED

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)   # ← ADDED
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        content = self.token_embed(x) + self.pos_embed(positions)   # (B, L, d)

        # Prepend CLS token — position 0 is now a dedicated classification token,
        # not a content coefficient.
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d)  ← ADDED
        h = torch.cat([cls, content], dim=1)             # (B, L+1, d)  ← ADDED

        h = self.encoder(h)
        h = self.norm(h[:, 0])   # CLS output — now truly position 0 = classification token
        return self.head(h)
```

**Note:** The paper does not specify the readout mechanism. This CLS-token approach (BERT-style) is architecturally principled and the most common choice for sequence classification with Transformers, but it is an implementation decision, not a paper specification.

---

### Mismatch B — Transformer nhead=12 is mathematically impossible

**Paper §4.3:** "12 attention heads, hidden dim 512"

`d_model=512` requires `d_model % nhead == 0` for integer head dimensions. `512 % 12 = 8 ≠ 0`. This is mathematically invalid for standard PyTorch `MultiheadAttention`.

**Also inconsistent:** The paper simultaneously claims ~51M parameters. For d\_model=512, 8 layers, ff=2048:
```
token_embed:  3329 × 512            =   1,704,448
pos_embed:    1536 × 512            =     786,432
8 × encoder layer:
  attention:  4 × 512²              =   1,048,576
  FFN:        2 × 512 × 2048        =   2,097,152
  norms:      4 × 512               =       2,048
  per layer total:                  =   3,147,776
8 layers:                           =  25,182,208
head (LayerNorm + Linear):          =     263,169
Total:                              ≈  27,936,257  (~27.9M)
```

To reach 51M with d\_model=512 would require ff\_dim ≈ 5,000 — not stated anywhere. The paper's parameter count appears to be a reporting error.

**Implementation decision:** Use `nhead=8` (head\_dim=64, clean integer division). This is the only architecturally valid choice for d\_model=512.

---

### Mismatch C — GNN parameter count: 18M claimed vs ~0.66M actual

**Paper §4.3:** "Total 18M parameters"

For SAGEConv(in→out), PyTorch Geometric implements two Linear layers: `lin_l` (in→out) and `lin_r` (in→out).

```
Layer 1 (in=1, out=256):
  lin_l: 1×256 + 256 = 512
  lin_r: 1×256 + 256 = 512
  total: 1,024

Layers 2–6 (in=256, out=256), ×5:
  lin_l: 256×256 + 256 = 65,792
  lin_r: 256×256 + 256 = 65,792
  total per layer: 131,584
  5 layers: 657,920

BatchNorm × 6: 6 × (2×256) = 3,072
head Linear(256,1): 256 + 1 = 257

Grand total: 1,024 + 657,920 + 3,072 + 257 ≈ 662,273 (~0.66M)
```

The paper's 18M claim is ~27× off for the stated architecture. Like the Transformer parameter count, this appears to be a paper error. The implementation's ~0.66M is the correct count for SAGEConv(1→256) × 6.

---

## 5. Cosmetic Fixes

### Fix 5 — datasets.py docstring: float32 → int16

**Old docstring:**
```python
"""
Each shard file must contain:
  a:     float32, shape (shard_size, k, n)   ← WRONG
  b:     float32, shape (shard_size, n)       ← WRONG
"""
```

**Fixed docstring:**
```python
"""
Each shard file must contain:
  a:     int16,   shape (shard_size, k, n)   — coefficients in [0, q)
  b:     int16,   shape (shard_size, n)      — coefficients in [0, q)
  label: int8,    shape (shard_size,)        — 1=LWE, 0=uniform
"""
```

The actual `save_shard()` call has always used `int16`:
```python
def save_shard(path, a, b, labels):
    np.savez_compressed(path,
                        a=a.astype(np.int16),      # was always int16
                        b=b.astype(np.int16),
                        label=labels.astype(np.int8))
```

---

### Fix 6 — Preflight check: split dirs defined before preflight

**Old order:**
- cell-10: defines `PARAM_SET`, `MODEL`, `BASE_DIR`, `DATA_DIR`, `CKPT_DIR`, `ckpt_out`, `best_model_path` — but NOT `train_dir/val_dir/test_dir`
- cell-11: preflight checks `train_dir`, `val_dir`, `test_dir` → always shows "MISSING" on fresh run
- cell-13: defines `train_dir`, `val_dir`, `test_dir`

**Fix — notebook cell-10 now includes:**
```python
# Split directories — defined here so the preflight check (next cell) can see them
train_dir = f"{DATA_DIR}/train"
val_dir   = f"{DATA_DIR}/val"
test_dir  = f"{DATA_DIR}/test"
```

---

### Fix 7 — Paper comparison cell: wrong AUROC values

**Old cell-29:**
```python
PAPER_RESULTS = {
    "ML-KEM-512": {
        "AUROC": 0.505,       # ← WRONG (not in paper)
        "Advantage": 0.010    # ← no paper source
    }
}
```

**Fixed cell-29** — values sourced directly from paper Table 2 (§6.2):
```python
PAPER_RESULTS = {
    "ML-KEM-512":  {"Transformer": 0.502, "GNN": 0.501, "Stat_baseline": 0.500},
    "ML-KEM-768":  {"Transformer": 0.500, "GNN": 0.500, "Stat_baseline": 0.500},
    "ML-KEM-1024": {"Transformer": 0.500, "GNN": 0.500, "Stat_baseline": 0.500},
}
paper_auroc = paper["Transformer"] if MODEL == "transformer" else paper["GNN"]
```

Also added ML-KEM-768 and ML-KEM-1024 entries (previously only 512 was present), and the comparison now dynamically selects Transformer vs GNN AUROC based on the `MODEL` variable.

---

## 6. Complete Fix List

| # | File | Symbol / Cell | What changed | Severity |
|---|------|--------------|--------------|----------|
| 1 | `src/latticeprobe/models/gnn.py` | `LWEGNN.__init__`, `forward` | GATConv → SAGEConv; removed edge\_attr from forward | **Critical** |
| 2 | `notebook/Lattice_Probe_Experiments.ipynb` | cell-13 | Removed `--secret-file` from val and test generation | **Critical** |
| 3 | `notebook/Lattice_Probe_Experiments.ipynb` | cell-15 | Added `--batch-size 32` (was OOM at 256) | **Critical** |
| 4 | `scripts/train.py` | `build_model()`, `__main__` | LWEGNN import moved lazy into GNN branch | **Critical** |
| 5 | `scripts/evaluate.py` | `load_checkpoint()`, `__main__` | LWEGNN import moved lazy into GNN branch | **Critical** |
| 6 | `src/latticeprobe/models/transformer.py` | `__init__`, `_init_weights`, `forward` | Added cls\_token parameter; prepend in forward | **Moderate** |
| 7 | `notebook/Lattice_Probe_Experiments.ipynb` | cell-10 | train/val/test\_dir defined before preflight | **Moderate** |
| 8 | `notebook/Lattice_Probe_Experiments.ipynb` | cell-29 | AUROC corrected from 0.505 → 0.502 (paper Table 2) | **Moderate** |
| 9 | `src/latticeprobe/datasets.py` | module docstring | Dtype: float32 → int16 | Cosmetic |

> **Checkpoint compatibility note:** All `best.pt` files saved before fixes 1 and 6 are **incompatible** with the updated model classes — they contain state dicts from GATConv (different layer names and shapes) and from the CLS-token-less Transformer. Retrain from scratch.

---

## 7. Codebase Architecture

### File Map

```
src/latticeprobe/
├── ring.py             ← NTT arithmetic (FIPS-203 §4.3 exact)
├── params.py           ← Parameter sets (LWEParams dataclass)
├── prng.py             ← Fresh per-sample ChaCha20-based RNG
├── sampler.py          ← LWE / uniform sample generators
├── representations.py  ← Sequence (int64 tokens) and graph (bipartite) encodings
├── datasets.py         ← PyTorch Dataset wrappers — sharded int16 .npz
├── baselines.py        ← LR, MLP, χ² distinguisher, lattice estimator, bootstrap AUROC
└── models/
    ├── transformer.py  ← LWETransformer — 8-layer, CLS-token, ~27M params
    └── gnn.py          ← LWEGNN — GraphSAGE 6-layer, ~0.66M params

scripts/
├── generate_dataset.py        ← Shard LWE/uniform data → .npz files
├── train.py                   ← Full training loop: AMP fp16, AdamW, cosine LR
├── evaluate.py                ← Inference + bootstrap AUROC + LR/MLP/χ² baselines
├── bit_recovery.py            ← Per-bit secret recovery experiment
├── sweep_sample_efficiency.py ← Sample complexity sweep N=2^10..2^20
└── analyze_scaling.py         ← Post-sweep analysis and plotting

notebooks/
├── LatticeProbe_Colab.ipynb              ← Paper pipeline (sanity checks + main eval)
└── notebook/
    └── Lattice_Probe_Experiments.ipynb   ← Extended experiments notebook
```

---

## 8. Mathematical Foundation

**Ring:** `R_q = Z_q[X] / (X^n + 1)`, `q = 3329`, `n = 256` (FIPS-203 exact)

**Module-LWE sample:** `b = <a, s> + e mod q`

| Symbol | Shape | Distribution |
|--------|-------|-------------|
| `a` | `(k, n)` | Uniform in `R_q^k` |
| `s` | `(k, n)` | CBD(η) or binary/ternary |
| `e` | `(n,)` | CBD(η) or Gaussian(σ) |
| `b` | `(n,)` | Derived; indistinguishable from uniform (LWE hardness) |

**NTT ring multiplication** (`src/latticeprobe/ring.py`):
```
poly_mul(f, g) = intt(ntt_mul(ntt(f), ntt(g)))
```
Follows FIPS-203 §4.3 exactly: `ZETAS[k] = 17^BitRev7(k) mod 3329`, `_INTT_FACTOR = 128^{-1} mod 3329 = 3303`.

---

## 9. Parameter Sets

**Standardised — FIPS-203 (paper §2.1.3):**

| Name | k | η | NIST Level | Transformer AUROC | GNN AUROC |
|------|---|---|------------|------------------|-----------|
| ML-KEM-512 | 2 | 3 | 1 (≈AES-128) | 0.502 | 0.501 |
| ML-KEM-768 | 3 | 2 | 3 (≈AES-192) | 0.500 | 0.500 |
| ML-KEM-1024 | 4 | 2 | 5 (≈AES-256) | 0.500 | 0.500 |

**Weakened — paper §5.4:**

| Name | Description (verbatim from paper) | σ | AUROC @ 2^16 |
|------|-----------------------------------|---|--------------|
| W1 | "binary secret, σ matching ML-KEM-512" | CBD(η=3) | 0.71 |
| W2 | "σ=0, no noise" | 0 | 1.00 |
| W3 | "ML-KEM-512 with σ reduced by 60%" | 0.490 | 0.83 |
| edge | "ML-KEM-512 with σ reduced 35%" (§4.1) | 0.796 | — (0.541 @ 2^18) |

σ derivation: `σ_512 = std(CBD(η=3)) = sqrt(η/2) = sqrt(1.5) ≈ 1.2247`. W3: `1.2247 × 0.40 = 0.490`. Edge: `1.2247 × 0.65 = 0.796`.

---

## 10. Data Representations

### Sequence — paper §4.2.1

```
(a, b) → flatten([a_0, a_1, ..., a_{k-1}, b]) → int64 tensor, length = k*n + n
```

Paper explicitly compares two encodings; **modular [0,q) "performs slightly better" and is used for reported results.** Implementation uses `to_sequence()` → `torch.long` tokens in `[0, 3329)` → MATCH.

| Param Set | seq\_len | Encoder input (with CLS) |
|-----------|---------|--------------------------|
| ML-KEM-512 | 768 | 769 |
| ML-KEM-768 | 1024 | 1025 |
| ML-KEM-1024 | 1280 | 1281 |

### Graph — paper §4.2.2

```
Bipartite graph:
  k*n variable nodes — feature = a[r,c] / q
  n   equation nodes — feature = b[c] / q
  2*k*n undirected edges — weight = a[r,c] / q (stored but ignored by SAGEConv)
```

### Dataset shards

```python
# src/latticeprobe/datasets.py  save_shard()
np.savez_compressed(path,
                    a=a.astype(np.int16),       # coefficients in [0, q)
                    b=b.astype(np.int16),
                    label=labels.astype(np.int8))
```

---

## 11. Model Architectures (Current, Post-Fix)

### LWETransformer

```python
# src/latticeprobe/models/transformer.py

class LWETransformer(nn.Module):
    def __init__(self, params, d_model=512, nhead=8, num_layers=8,
                 dim_feedforward=2048, dropout=0.1):
        self.token_embed = nn.Embedding(q=3329, d_model)       # 1.70M params
        self.pos_embed   = nn.Embedding(seq_len*2, d_model)    # 0.79M params
        self.cls_token   = nn.Parameter(zeros(1, 1, d_model))  # 0.001M params
        self.encoder     = TransformerEncoder(8 layers)         # 25.18M params
          # each layer: nhead=8, dim_ff=2048, norm_first=True, batch_first=True
        self.norm        = LayerNorm(d_model)
        self.head        = Linear(d_model, 1)
        # Total: ~27.9M params

    def forward(self, x):                      # x: (B, seq_len) int64
        content = token_embed(x) + pos_embed(positions)   # (B, L, 512)
        cls = cls_token.expand(B, 1, 512)                 # (B, 1, 512)
        h = cat([cls, content], dim=1)                    # (B, L+1, 512)
        h = encoder(h)                                    # (B, L+1, 512)
        return head(norm(h[:, 0]))                        # (B, 1) via CLS
```

### LWEGNN

```python
# src/latticeprobe/models/gnn.py

class LWEGNN(nn.Module):
    def __init__(self, params, hidden=256, num_layers=6, dropout=0.1):
        self.convs = [SAGEConv(in_ch, hidden) for each layer]   # GraphSAGE
        self.norms = [BatchNorm1d(hidden) for each layer]
        self.head  = Linear(hidden, 1)
        # Total: ~0.66M params

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)   # mean-aggregate neighbours (no edge_attr)
            x = norm(x).relu()
        x = global_mean_pool(x, batch)   # (B, 256)
        return self.head(x)              # (B, 1)
```

---

## 12. Training Pipeline

**Paper §4.4 hyperparameters — all matched:**

```python
# scripts/train.py defaults
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
# epochs=50, patience=5, gradient_clip=1.0, mixed_precision=fp16
```

**T4 batch size:**
- Paper trained on H100-class hardware; batch size not specified
- T4 (15 GB VRAM): `--batch-size 32` required at seq\_len=769
- H100 (80 GB VRAM): `--batch-size 256` is feasible

**Training sample scale:**

| Context | Train samples | Flag |
|---------|--------------|------|
| Paper (§4.4) | 2^20 = 1,048,576 | `--n-samples 1048576` |
| Notebook demo | 2^14 = 16,384 | current `TRAIN_SAMPLES = 16384` |

---

## 13. Evaluation Pipeline

```
best.pt
  ↓
load_checkpoint()          ← lazy LWEGNN import; loads saved args for arch params
  ↓
_inference()               ← no_grad, fp32 output
  ↓
bootstrap_auroc()          ← BCa bootstrap, 100 resamples, 95% CI (paper §5.2)
chi2_distinguisher()       ← per-coefficient deviation from q/2 (no training data)
run_logistic_regression()  ← sklearn LR on flattened (a,b)
run_mlp()                  ← sklearn MLP(512,256) with early stopping
  ↓
results.json
```

---

## 14. Data Flow (End-to-End)

```
LWEParams (params.py)
    │
    ▼
sampler.generate_batch()
    │  CBD / Gaussian noise; NTT poly_mul for b = <a,s>+e mod q
    ▼
datasets.save_shard()              ← int16 a/b, int8 labels → shard_NNNNN.npz
    │
    ▼
generate_dataset.py                ← ONE FRESH SECRET per split (paper §5.2)
    │  train secret ≠ val secret ≠ test secret
    ▼
LWESequenceDataset  or  LWEGraphDataset
    │  Sequence: to_sequence() → int64 tokens in [0,3329)
    │  Graph:    to_graph()    → bipartite Data; SAGEConv ignores edge_attr
    ▼
DataLoader → train.py run_epoch()
    │  Transformer: CLS prepend → 8-layer encoder → h[:,0] → head
    │  GNN:         SAGEConv×6 → global_mean_pool → head
    │  AdamW + AMP fp16 + GradScaler + CosineAnnealingLR
    │  batch_size=32 on T4 / batch_size=256 on H100
    ▼
best.pt  (model_state + args dict)
    │
    ▼
evaluate.py evaluate_model()
    │  bootstrap AUROC CI + LR/MLP/χ² baselines
    ▼
results.json  +  compute_log.csv
```

---

## 15. Paper Inconsistencies (Cannot Be Fixed)

These are errors or omissions in the paper itself — they cannot be resolved without author clarification:

| Issue | Paper states | Correct value | Analysis |
|-------|-------------|---------------|----------|
| Transformer nhead | 12 | 8 | 512 % 12 ≠ 0; mathematically invalid |
| Transformer params | ~51M | ~27.9M | d\_model=512, 8 layers, ff=2048 = 27.9M; reaching 51M requires ff≈5000 |
| GNN params | ~18M | ~0.66M | SAGEConv(1→256)×6 + BN + head = 662,273 params |
| CLS token / readout | Not specified | CLS-token (BERT-style) added | Implementation decision; not verifiable from paper |
| FFN inner dim | Not specified | 2048 (4×d\_model standard) | Reasonable default; paper omits it |

---

## 16. All Issues — Final Status

| Issue | Severity | Status |
|-------|----------|--------|
| GNN backbone: GATConv instead of GraphSAGE | Critical | ✅ FIXED |
| Val/test sharing train secret (violates paper §5.2) | Critical | ✅ FIXED |
| Transformer training OOM on T4 (batch=256) | Critical | ✅ FIXED (batch=32) |
| `train.py`: unconditional LWEGNN import crashed transformer training | Critical | ✅ FIXED |
| `evaluate.py`: unconditional LWEGNN import crashed transformer eval | Critical | ✅ FIXED |
| Transformer CLS token missing (pos 0 was a content token) | Moderate | ✅ FIXED |
| Paper comparison: AUROC 0.505 wrong (correct: 0.502 from Table 2) | Moderate | ✅ FIXED |
| Preflight check: train/val/test\_dir undefined at check time | Moderate | ✅ FIXED |
| `datasets.py` docstring: float32 → int16 | Cosmetic | ✅ FIXED |
| Transformer nhead=12 impossible with d\_model=512 | Paper error | ⚠️ Cannot fix — 8 is the only valid choice |
| Transformer ~51M param claim (actual ~27.9M) | Paper error | ⚠️ Cannot fix |
| GNN ~18M param claim (actual ~0.66M) | Paper error | ⚠️ Cannot fix |
| Training scale: paper 2^20, demo 2^14 | By design | ℹ️ Document only |
| CLS pooling strategy not specified in paper | Unspecified | ℹ️ Architecturally principled; flagged |
| Dual-domain repr (`--repr dual/ntt`) not in paper | Extension | ℹ️ Flagged |
| Sections 8–10 in Experiments notebook are placeholder stubs | Open | ✅ Flagged with TODO markers |
| Checkpoint fingerprinting: architecture_version missing | Moderate | ✅ FIXED |
| No checkpoint compat check on load | Moderate | ✅ FIXED |
| No automated architecture consistency tests | Moderate | ✅ FIXED (tests/test_architecture.py) |
| README claims "~51M-parameter models" | Documentation | ✅ FIXED |
| README missing Paper Inconsistencies section | Documentation | ✅ FIXED |
| REPORT.md not generated | Deliverable | ✅ GENERATED |

---

## 17. Final Round Fixes (REVIEW.md §7–10)

### Fix 8 — Checkpoint fingerprinting and backward-compat detection

**Files:** `scripts/train.py`, `scripts/evaluate.py`

Every checkpoint now saves two metadata keys:
```python
torch.save({
    "architecture_version": "v2-reconciled",   # ← NEW
    "model_type": args.model,                   # ← NEW (e.g. "transformer")
    "epoch": epoch,
    "val_auroc": val_auroc,
    "model_state": model.state_dict(),
    "args": vars(args),
}, best_ckpt)
```

At load time, `_check_checkpoint_compat()` in `evaluate.py` detects stale checkpoints:

```python
def _check_checkpoint_compat(ckpt, model_name, path):
    arch_ver = ckpt.get("architecture_version", None)
    state = ckpt.get("model_state", {})

    if arch_ver is None:
        if model_name == "gnn":
            gat_keys = [k for k in state if "att_src" in k or "att_dst" in k or "lin_edge" in k]
            if gat_keys:
                raise RuntimeError(
                    f"Checkpoint '{path}' was trained with GATConv (pre-reconciliation). "
                    "Retrain with: python scripts/train.py --model gnn ..."
                )
        if model_name == "transformer":
            if not any("cls_token" in k for k in state):
                raise RuntimeError(
                    f"Checkpoint '{path}' predates the CLS-token architecture fix. "
                    "Retrain with: python scripts/train.py --model transformer ..."
                )
```

Detection heuristics:
- **Old GNN (GATConv):** state dict contains `att_src` / `att_dst` / `lin_edge` keys (GATConv-specific)
- **Old Transformer (no CLS):** state dict has no `cls_token` key

---

### Fix 9 — `validate_disjoint_secrets()` in generate_dataset.py

```python
def validate_disjoint_secrets(*secret_paths: str) -> None:
    """Assert that all provided secrets.npy files contain distinct secrets."""
    import hashlib
    hashes = {}
    for path in secret_paths:
        digest = hashlib.sha256(np.load(path).tobytes()).hexdigest()
        if digest in hashes:
            raise RuntimeError(
                f"Secret reuse detected between '{hashes[digest]}' and '{path}'. "
                "This violates paper §5.2 (disjoint key sets)."
            )
        hashes[digest] = path
```

Also: `--secret-file` CLI flag now emits `warnings.warn()` to alert researchers.

---

### Fix 10 — count_parameters() added to both models

Both `LWETransformer` and `LWEGNN` now expose:
```python
def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

These are also used in `tests/test_architecture.py` to verify exact parameter counts.

---

### Fix 11 — README.md: removed "~51M-parameter models" claim

Old text:
```
...even with up to 2²⁰ training samples and ~51M-parameter models...
```

New text:
```
...even with up to 2²⁰ training samples...
```

A new "Paper Inconsistencies" section was added to the README with a table of all five discrepancies (nhead, transformer params, GNN params, CLS token, FFN dim).

---

### Fix 12 — Notebook sections 8–10 marked as OPEN ISSUE

Sections 8 (Noise Phase Transition), 9 (Secret Diversity Scaling), and 10 (Sample Complexity Scaling) in `notebook/Lattice_Probe_Experiments.ipynb` were stub cells that silently re-ran training. They now display:

```
OPEN ISSUE: Section 8 (Noise Phase Transition) not yet implemented.
See REVIEW.md §10 for requirements.
```

Each section also has a markdown cell above describing what the TODO requires.

---

### Fix 13 — tests/test_architecture.py (new, 15 tests)

New test file verifying all critical architecture invariants:

| Test | What it checks |
|------|----------------|
| `test_transformer_parameter_count` | ~27,936,257 ±5% (paper claims 51M — PAPER INCONSISTENCY) |
| `test_gnn_parameter_count` | ~662,273 ±5% (paper claims 18M — PAPER INCONSISTENCY) |
| `test_cls_token_present` | `cls_token` is `nn.Parameter` with shape `(1,1,512)` |
| `test_cls_token_prepended` | Encoder receives seq_len+1 tokens (CLS + content) |
| `test_cls_token_initialised` | `cls_token.abs().sum() > 0` after `_init_weights` |
| `test_disjoint_secret_generation` | Two independent calls produce distinct `secrets.npy` |
| `test_validate_disjoint_secrets_raises_on_reuse` | `RuntimeError` when same path given twice |
| `test_validate_disjoint_secrets_passes_for_distinct` | No error for genuinely distinct secrets |
| `test_lazy_gnn_import_train` | AST: `LWEGNN` not in top-level imports of train.py |
| `test_lazy_gnn_import_eval` | AST: `LWEGNN` not in top-level imports of evaluate.py |
| `test_sequence_lengths[ML-KEM-512]` | seq=768, encoder_input=769 |
| `test_sequence_lengths[ML-KEM-768]` | seq=1024, encoder_input=1025 |
| `test_sequence_lengths[ML-KEM-1024]` | seq=1280, encoder_input=1281 |
| `test_checkpoint_architecture_version` | AST: "v2-reconciled" appears ≥2× in train.py torch.save calls |
| `test_checkpoint_compat_rejects_gat` | `RuntimeError("GATConv")` on old state dict |
| `test_checkpoint_compat_rejects_pre_cls_transformer` | `RuntimeError("CLS-token")` on no-cls state dict |
| `test_no_gatconv_in_source` | GATConv absent from all files in src/ |

Run with:
```bash
pytest tests/test_architecture.py -v
```
