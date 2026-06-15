# Reproducing Paper Results

This document gives the exact commands needed to regenerate every table and figure in Ologunde (2026). Run all commands from the repository root unless stated otherwise.

---

## Prerequisites

```bash
git clone https://github.com/23520904/Lattice-Probe.git
cd Lattice-Probe
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pytest tests/ -q   # should report 62+ passed
```

Alternatively, use the Docker image for a fully pinned environment:

```bash
docker build -t latticeprobe:latest .
docker run --rm --gpus all -v $(pwd)/data:/workspace/data latticeprobe:latest bash
```

---

## Step 1 — Generate datasets

Each parameter set needs train / val / test splits with **disjoint secrets**.
Dataset sizes used in the paper: 2¹⁶ = 65 536 (small), 2¹⁸ = 262 144 (main), 2²⁰ = 1 048 576 (scaling).

```bash
for SPLIT in train val test; do
  python scripts/generate_dataset.py \
      --param-set ML-KEM-512  --n-samples 262144 --output-dir data/ML-KEM-512/$SPLIT
  python scripts/generate_dataset.py \
      --param-set ML-KEM-768  --n-samples 262144 --output-dir data/ML-KEM-768/$SPLIT
  python scripts/generate_dataset.py \
      --param-set ML-KEM-1024 --n-samples 262144 --output-dir data/ML-KEM-1024/$SPLIT
  python scripts/generate_dataset.py \
      --param-set W1   --n-samples 65536 --output-dir data/W1/$SPLIT
  python scripts/generate_dataset.py \
      --param-set W2   --n-samples 65536 --output-dir data/W2/$SPLIT
  python scripts/generate_dataset.py \
      --param-set W3   --n-samples 65536 --output-dir data/W3/$SPLIT
  python scripts/generate_dataset.py \
      --param-set edge --n-samples 262144 --output-dir data/edge/$SPLIT
done
```

> Each split call draws a **fresh random secret**, ensuring train/val/test keys are disjoint as required by paper §5.2.

---

## Step 2 — Sanity checks (W1 / W2 / W3)

Train on weakened regimes first. If these don't show AUROC ≫ 0.5, there is a methodology bug — **do not proceed to standardised params** until these pass.

```bash
# W2 (no noise) — should reach AUROC ≈ 1.0 within a few epochs
python scripts/train.py \
    --param-set W2 --model transformer \
    --train-dir data/W2/train --val-dir data/W2/val \
    --output-dir checkpoints/transformer-W2 --epochs 10

# W1 (binary secret)
python scripts/train.py \
    --param-set W1 --model transformer \
    --train-dir data/W1/train --val-dir data/W1/val \
    --output-dir checkpoints/transformer-W1 --epochs 20

# W3 (σ reduced 60%)
python scripts/train.py \
    --param-set W3 --model transformer \
    --train-dir data/W3/train --val-dir data/W3/val \
    --output-dir checkpoints/transformer-W3 --epochs 30
```

---

## Step 3 — Train Transformer on standardised parameters

```bash
for PSET in ML-KEM-512 ML-KEM-768 ML-KEM-1024; do
  python scripts/train.py \
      --param-set $PSET --model transformer \
      --train-dir data/$PSET/train \
      --val-dir   data/$PSET/val \
      --output-dir checkpoints/transformer-$PSET \
      --epochs 50 --batch-size 256 --lr 1e-4 --patience 5
done
```

With W&B logging (optional):

```bash
python scripts/train.py \
    --param-set ML-KEM-512 --model transformer \
    --train-dir data/ML-KEM-512/train \
    --val-dir   data/ML-KEM-512/val \
    --output-dir checkpoints/transformer-512 \
    --wandb --wandb-project latticeprobe
```

---

## Step 4 — Train GNN on standardised parameters + edge-of-margin

```bash
for PSET in ML-KEM-512 ML-KEM-768 ML-KEM-1024 edge; do
  python scripts/train.py \
      --param-set $PSET --model gnn \
      --train-dir data/$PSET/train \
      --val-dir   data/$PSET/val \
      --output-dir checkpoints/gnn-$PSET \
      --epochs 50 --batch-size 256 --lr 1e-4 --patience 5
done
```

---

## Step 5 — Evaluate (Table 2)

```bash
mkdir -p results

for PSET in ML-KEM-512 ML-KEM-768 ML-KEM-1024; do
  python scripts/evaluate.py \
      --checkpoint checkpoints/transformer-$PSET/best.pt \
      --model transformer --param-set $PSET \
      --test-dir  data/$PSET/test \
      --train-dir data/$PSET/train \
      --n-boot 100 \
      --output-json results/transformer-$PSET.json

  python scripts/evaluate.py \
      --checkpoint checkpoints/gnn-$PSET/best.pt \
      --model gnn --param-set $PSET \
      --test-dir  data/$PSET/test \
      --train-dir data/$PSET/train \
      --n-boot 100 \
      --output-json results/gnn-$PSET.json
done

# Edge-of-margin GNN
python scripts/evaluate.py \
    --checkpoint checkpoints/gnn-edge/best.pt \
    --model gnn --param-set edge \
    --test-dir  data/edge/test \
    --train-dir data/edge/train \
    --n-boot 100 \
    --output-json results/gnn-edge.json
```

---

## Step 6 — Partial-bit recovery experiment

```bash
for PSET in ML-KEM-512 ML-KEM-768 ML-KEM-1024; do
  python scripts/bit_recovery.py \
      --train-dir data/$PSET/train \
      --test-dir  data/$PSET/test \
      --param-set $PSET \
      --output-json results/bit_recovery-$PSET.json
done
```

Expected result: mean per-bit accuracy ≈ 0.500 for all standardised parameter sets.

---

## Step 7 — Compute scaling experiment

Generate datasets at 2¹⁴, 2¹⁶, 2¹⁸ and train the GNN (or Transformer) on ML-KEM-512:

```bash
for N in 16384 65536 262144; do
  python scripts/generate_dataset.py \
      --param-set ML-KEM-512 --n-samples $N \
      --output-dir data/scaling/ML-KEM-512-n$N/train
  python scripts/generate_dataset.py \
      --param-set ML-KEM-512 --n-samples $((N/8)) \
      --output-dir data/scaling/ML-KEM-512-n$N/val

  python scripts/train.py \
      --param-set ML-KEM-512 --model gnn \
      --train-dir data/scaling/ML-KEM-512-n$N/train \
      --val-dir   data/scaling/ML-KEM-512-n$N/val \
      --output-dir checkpoints/scaling/gnn-n$N \
      --epochs 50 --patience 5 \
      --compute-log compute_log.csv
done
```

GPU-hours are recorded in `compute_log.csv` per epoch.

---

## Expected results (Table 2)

| Parameter | Transformer AUROC | GNN AUROC | χ² baseline |
|---|---|---|---|
| ML-KEM-512 | 0.502 | 0.501 | 0.500 |
| ML-KEM-768 | 0.500 | 0.500 | 0.500 |
| ML-KEM-1024 | 0.500 | 0.500 | 0.500 |
| edge (2¹⁸) | — | 0.541 | 0.503 |

All values are means over 100 bootstrap resamples. Deviations within ±0.005 are expected from run-to-run randomness.

---

## Verifying the test suite

```bash
pytest tests/ -v
# Expected: 62+ tests passed in < 3 minutes on CPU
```

Individual test modules:

```bash
pytest tests/test_ring.py          # NTT correctness
pytest tests/test_sampler.py       # LWE structure + noise distributions
pytest tests/test_representations.py  # sequence + graph shapes
pytest tests/test_models.py        # model forward passes
pytest tests/test_pipeline.py      # end-to-end generate→train→evaluate
```

---

*Based on: Ologunde, E. "AI-Accelerated Cryptanalysis of Lattice-Based Schemes." 2026.*
