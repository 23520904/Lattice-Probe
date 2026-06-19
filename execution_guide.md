# Lattice-Probe Execution Guide

Extracted strictly from the current codebase.

---

# Part 1 — CLI Reference

## Script
```text
scripts/generate_dataset.py
```
## Usage
```bash
python scripts/generate_dataset.py --help
```
## Arguments
| Argument | Type | Default | Required | Description |
|----------|--------|----------|----------|----------|
| `--param-set` | string | - | Yes | Parameter set (e.g. ML-KEM-512, W3, edge) |
| `--n-samples` | int | - | Yes | Total samples to generate (e.g. 65536 = 2^16) |
| `--shard-size` | int | 4096 | No | Samples per .npz shard file |
| `--output-dir` | string | - | Yes | Directory to write shard_*.npz and secret.npy |
| `--num-secrets` | int | 1 | No | Number of distinct secrets to generate and embed |
| `--secret-file` | string | None | No | Path to an existing secrets.npy file to reuse |
| `--noise-scale` | float | 1.0 | No | Scale multiplier for the noise variance |
| `--quiet` | flag | False | No | Suppress progress bar |

## Script
```text
scripts/train.py
```
## Usage
```bash
python scripts/train.py --help
```
## Arguments
| Argument | Type | Default | Required | Description |
|----------|--------|----------|----------|----------|
| `--param-set` | string | - | Yes | Parameter set |
| `--model` | string | - | Yes | transformer or gnn |
| `--train-dir` | string | - | Yes | Training data directory |
| `--val-dir` | string | - | Yes | Validation data directory |
| `--output-dir` | string | - | Yes | Checkpoint directory |
| `--epochs` | int | 50 | No | Training epochs |
| `--batch-size` | int | 256 | No | Batch size |
| `--lr` | float | 1e-4 | No | Learning rate |
| `--weight-decay` | float | 1e-2 | No | Weight decay |
| `--patience` | int | 5 | No | Early-stopping patience |
| `--ckpt-every` | int | 5 | No | Save checkpoint every N epochs |
| `--device` | string | auto | No | cuda / cpu / auto |
| `--d-model` | int | 512 | No | Transformer d_model |
| `--nhead` | int | 8 | No | Transformer nhead |
| `--num-layers` | int | 8 | No | Transformer layers |
| `--ff-dim` | int | 2048 | No | Transformer FFN inner dim |
| `--hidden` | int | 256 | No | GNN hidden dim |
| `--gnn-layers` | int | 6 | No | GNN layers |
| `--wandb` | flag | False | No | Enable W&B logging |
| `--wandb-project` | string | latticeprobe | No | W&B project |
| `--compute-log` | string | compute_log.csv | No | CSV for GPU-hour tracking |
| `--shuffle-labels` | flag | False | No | Randomly permute training labels |
| `--repr` | string | coeff | No | Representation domain (coeff, ntt, dual) |

## Script
```text
scripts/evaluate.py
```
## Usage
```bash
python scripts/evaluate.py --help
```
## Arguments
| Argument | Type | Default | Required | Description |
|----------|--------|----------|----------|----------|
| `--checkpoint` | string | - | Yes | Path to checkpoint |
| `--model` | string | - | Yes | transformer or gnn |
| `--param-set` | string | - | Yes | Parameter set |
| `--test-dir` | string | - | Yes | Test data directory |
| `--train-dir` | string | None | No | Training dir (for LR/MLP baselines) |
| `--batch-size` | int | 256 | No | Batch size |
| `--n-boot` | int | 100 | No | Bootstrap resamples for CI |
| `--device` | string | auto | No | cuda / cpu / auto |
| `--output-json` | string | None | No | Write results to JSON |
| `--shuffle-labels` | flag | False | No | Permute labels for testing |
| `--repr` | string | coeff | No | Representation domain (coeff, ntt, dual) |

## Script
```text
scripts/bit_recovery.py
```
```text
Not found in codebase
```

---

# Part 2 — Available Parameter Sets

| Name | n | k | q | Noise Distribution | Notes |
|--------|--------|--------|--------|--------|--------|
| ML-KEM-512 | 256 | 2 | 3329 | cbd (eta=3) | Standard FIPS-203 |
| ML-KEM-768 | 256 | 3 | 3329 | cbd (eta=2) | Standard FIPS-203 |
| ML-KEM-1024 | 256 | 4 | 3329 | cbd (eta=2) | Standard FIPS-203 |
| W1 | 256 | 2 | 3329 | cbd (eta=3) | binary secret |
| W2 | 256 | 2 | 3329 | zero | no noise |
| W3 | 256 | 2 | 3329 | gaussian (sigma=0.490) | σ reduced 60% below ML-KEM-512 |
| CBD-eta2 | 256 | 2 | 3329 | cbd (eta=2) | Tier A: CBD vs Gaussian |
| CBD-eta3 | 256 | 2 | 3329 | cbd (eta=3) | Tier A: CBD vs Gaussian |
| Gauss-var1.0 | 256 | 2 | 3329 | gaussian (sigma=1.0) | Tier A: CBD vs Gaussian |
| Gauss-var1.5 | 256 | 2 | 3329 | gaussian (sigma=1.22474487)| Tier A: CBD vs Gaussian |
| Sec-Binary | 256 | 2 | 3329 | cbd (eta=3) | Tier B: binary secret |
| Sec-Ternary | 256 | 2 | 3329 | cbd (eta=3) | Tier B: ternary secret |
| Sec-Uniform | 256 | 2 | 3329 | cbd (eta=3) | Tier B: uniform secret |
| edge | 256 | 2 | 3329 | gaussian (sigma=0.796)| σ reduced 35% |

---

# Part 3 — Dataset Generation Commands

## Standard ML-KEM-512
```bash
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --output-dir data/ML-KEM-512/train
```

## Standard ML-KEM-768
```bash
python scripts/generate_dataset.py --param-set ML-KEM-768 --n-samples 1048576 --output-dir data/ML-KEM-768/train
```

## Standard ML-KEM-1024
```bash
python scripts/generate_dataset.py --param-set ML-KEM-1024 --n-samples 1048576 --output-dir data/ML-KEM-1024/train
```

## Edge-of-Margin Dataset
```bash
python scripts/generate_dataset.py --param-set edge --n-samples 1048576 --output-dir data/edge/train
```

## Weakened Dataset
```bash
python scripts/generate_dataset.py --param-set W3 --n-samples 1048576 --output-dir data/W3/train
```

---

# Part 4 — Training Commands

## Transformer
```bash
python scripts/train.py \
  --param-set ML-KEM-512 \
  --model transformer \
  --train-dir data/ML-KEM-512/train \
  --val-dir data/ML-KEM-512/val \
  --output-dir checkpoints/transformer-512 \
  --epochs 50 \
  --batch-size 256 \
  --lr 0.0001 \
  --weight-decay 0.01 \
  --d-model 512 \
  --nhead 8 \
  --num-layers 8 \
  --ff-dim 2048
```

## GNN
```bash
python scripts/train.py \
  --param-set ML-KEM-512 \
  --model gnn \
  --train-dir data/ML-KEM-512/train \
  --val-dir data/ML-KEM-512/val \
  --output-dir checkpoints/gnn-512 \
  --epochs 50 \
  --batch-size 256 \
  --lr 0.0001 \
  --weight-decay 0.01 \
  --hidden 256 \
  --gnn-layers 6
```

---

# Part 5 — Evaluation Commands

## Normal Evaluation
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/transformer-512/best.pt \
  --model transformer \
  --param-set ML-KEM-512 \
  --test-dir data/ML-KEM-512/test \
  --train-dir data/ML-KEM-512/train
```

## Permutation Test
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/transformer-512/best.pt \
  --model transformer \
  --param-set ML-KEM-512 \
  --test-dir data/ML-KEM-512/test \
  --train-dir data/ML-KEM-512/train \
  --shuffle-labels
```

## Cross-Secret Evaluation
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/transformer-512/best.pt \
  --model transformer \
  --param-set ML-KEM-512 \
  --test-dir data/ML-KEM-512/test_secret_B
```

## Bit Recovery
```bash
Not found in codebase
```

---

# Part 6 — Reproduce Paper Tables

## Table Name: Performance on ML-KEM-512

### Dataset Generation
```bash
# Generate train dataset (creates a fresh secret)
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --output-dir data/ML-KEM-512/train

# Generate val and test datasets reusing the exact same secret from the train split
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 8192 --output-dir data/ML-KEM-512/val --secret-file data/ML-KEM-512/train/secrets.npy
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 8192 --output-dir data/ML-KEM-512/test --secret-file data/ML-KEM-512/train/secrets.npy
```

### Training
```bash
python scripts/train.py --param-set ML-KEM-512 --model transformer --train-dir data/ML-KEM-512/train --val-dir data/ML-KEM-512/val --output-dir checkpoints/transformer-512
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/transformer-512/best.pt --model transformer --param-set ML-KEM-512 --test-dir data/ML-KEM-512/test --train-dir data/ML-KEM-512/train
```

### Expected Output Metrics
- AUROC
- Advantage
- BCa Confidence Interval
- Cohen's d
- Logit Separation

---

# Part 7 — New Research Experiments

## Noise Phase Transition

```bash
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 1.00 --output-dir data/ML-KEM-512/noise_1.00/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.95 --output-dir data/ML-KEM-512/noise_0.95/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.90 --output-dir data/ML-KEM-512/noise_0.90/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.85 --output-dir data/ML-KEM-512/noise_0.85/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.80 --output-dir data/ML-KEM-512/noise_0.80/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.75 --output-dir data/ML-KEM-512/noise_0.75/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.70 --output-dir data/ML-KEM-512/noise_0.70/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.65 --output-dir data/ML-KEM-512/noise_0.65/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.60 --output-dir data/ML-KEM-512/noise_0.60/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.55 --output-dir data/ML-KEM-512/noise_0.55/train
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --noise-scale 0.50 --output-dir data/ML-KEM-512/noise_0.50/train
```

## Cross-Secret Transfer

Train:
```bash
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --output-dir data/ML-KEM-512/secret_A
python scripts/train.py --param-set ML-KEM-512 --model transformer --train-dir data/ML-KEM-512/secret_A --val-dir data/ML-KEM-512/val --output-dir checkpoints/transformer_secret_A
```

Test:
```bash
# Note: secret_B is generated WITHOUT --secret-file, explicitly drawing a disjoint key.
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 8192 --output-dir data/ML-KEM-512/secret_B
python scripts/evaluate.py --checkpoint checkpoints/transformer_secret_A/best.pt --model transformer --param-set ML-KEM-512 --test-dir data/ML-KEM-512/secret_B
```

## Secret Diversity Scaling

```bash
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --num-secrets 1 --output-dir data/ML-KEM-512/sec_div_1
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --num-secrets 10 --output-dir data/ML-KEM-512/sec_div_10
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --num-secrets 100 --output-dir data/ML-KEM-512/sec_div_100
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --num-secrets 1000 --output-dir data/ML-KEM-512/sec_div_1000
```

## Sample Complexity Scaling

```bash
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1024 --output-dir data/ML-KEM-512/scale_1024
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 4096 --output-dir data/ML-KEM-512/scale_4096
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 16384 --output-dir data/ML-KEM-512/scale_16384
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 65536 --output-dir data/ML-KEM-512/scale_65536
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 262144 --output-dir data/ML-KEM-512/scale_262144
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 1048576 --output-dir data/ML-KEM-512/scale_1048576
python scripts/generate_dataset.py --param-set ML-KEM-512 --n-samples 4194304 --output-dir data/ML-KEM-512/scale_4194304
```
*(Alternatively, execute the fully automated script `python scripts/sweep_sample_efficiency.py` which handles this internally).*

---

# Part 8 — Colab Execution Plan

| Experiment | Estimated Runtime | GPU Required | Priority |
|------------|------------|------------|------------|
| Generate Base Datasets (Part 6) | 5 mins | No | 1 |
| Train Standard Transformer (Part 6)| 40 mins | Yes | 2 |
| Standard Evaluation (Part 6) | 2 mins | Yes | 3 |
| Sample Efficiency Sweep (`scripts/sweep_sample_efficiency.py`) | 20 mins | Yes | 4 |
| Feature Occlusion / Interpretability | 5 mins | Yes | 5 |
| Cross-Secret Transfer | 10 mins | Yes | 6 |
| Noise Phase Transition Suite | 120 mins | Yes | 7 |
| Secret Diversity Scaling Suite | 90 mins | Yes | 8 |
