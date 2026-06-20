# Generate a Complete Google Colab Notebook for Lattice-Probe

I want you to generate a complete Jupyter Notebook (`.ipynb`) for Google Colab.

The notebook should be self-contained, reproducible, and suitable for running large-scale Lattice-Probe experiments.

Do NOT generate markdown documentation only.

Generate actual notebook cells in the order they should appear.

---

# Requirements

The notebook must be organized into sections.

Each section should contain:

* Markdown explanation cell
* Executable code cell(s)
* Progress output
* Error checking
* Result saving

Use clean notebook formatting.

---

# Section 1 — Environment Setup

Create notebook cells that:

## Verify GPU

Display:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Fail gracefully if GPU is unavailable.

---

## Clone Repository

Provide a variable:

```python
REPO_URL = "REPLACE_ME"
```

and clone if repository does not exist.

---

## Install Dependencies

Install:

```text
numpy
scipy
scikit-learn
pandas
matplotlib
tqdm
pynacl
torch-geometric
```

and any missing dependencies found in requirements files.

---

## Verify CLI

Run:

```bash
python scripts/generate_dataset.py --help
python scripts/train.py --help
python scripts/evaluate.py --help
```

and stop if any command fails.

---

# Section 2 — Configuration

Create a notebook configuration cell.

Example:

```python
PARAM_SET = "ML-KEM-512"

TRAIN_SAMPLES = 131072 #1048576
VAL_SAMPLES = 8192
TEST_SAMPLES = 8192

MODEL = "transformer"

BASE_DIR = "experiments/mlkem512"
```

All later cells must use these variables.

Do not hardcode paths.

---

# Section 3 — Dataset Generation

Generate:

```text
train
val
test
```

datasets.

Use:

```bash
--secret-file
```

to ensure:

```text
Train Secret = Val Secret = Test Secret
```

Verify:

```python
os.path.exists(...)
```

after generation.

Display dataset sizes.

---

# Section 4 — Transformer Training

Train:

```bash
python scripts/train.py
```

using notebook configuration variables.

Save:

```text
best.pt
compute_log.csv
```

Display:

* training loss
* validation loss
* AUROC

if available in logs.

---

# Section 5 — Standard Evaluation

Run:

```bash
python scripts/evaluate.py
```

Store results in:

```python
results_standard.json
```

Parse JSON.

Display:

| Metric           | Value |
| ---------------- | ----- |
| AUROC            |       |
| Advantage        |       |
| Cohen d          |       |
| Logit Separation |       |

---

# Section 6 — Permutation Test

Run:

```bash
--shuffle-labels
```

training and evaluation.

Store:

```python
results_permutation.json
```

Display:

```text
Expected:
AUROC ≈ 0.5
Advantage ≈ 0
```

Highlight if violated.

---

# Section 7 — Cross Secret Evaluation

Generate:

```text
Secret A
Secret B
```

Train on:

```text
Secret A
```

Evaluate on:

```text
Secret B
```

Store:

```python
results_cross_secret.json
```

Create comparison table:

| Metric    | Same Secret | Cross Secret |
| --------- | ----------- | ------------ |
| AUROC     |             |              |
| Advantage |             |              |

---

# Section 8 — Noise Phase Transition Sweep

Automatically loop through:

```python
NOISE_SCALES = [
    1.00,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
]
```

For each scale:

1. Generate dataset
2. Train model
3. Evaluate model

Store results in:

```python
phase_transition.csv
```

Create:

```python
Advantage vs Noise Scale
```

plot.

Use matplotlib.

---

# Section 9 — Secret Diversity Scaling

Loop through:

```python
SECRET_COUNTS = [
    1,
    10,
    100,
    1000,
]
```

using:

```bash
--num-secrets
```

For each experiment:

1. Generate dataset
2. Train model
3. Evaluate model

Store:

```python
secret_diversity.csv
```

Create plot:

```python
Advantage vs Number of Secrets
```

Use log-scale x-axis.

---

# Section 10 — Sample Complexity Scaling

Run:

```python
SAMPLE_COUNTS = [
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
]
```

For each:

1. Generate dataset
2. Train
3. Evaluate

Store:

```python
sample_efficiency.csv
```

Plot:

```python
Advantage vs Samples
```

Use logarithmic x-axis.

---

# Section 11 — Paper Comparison

Create a section where I can manually enter paper values.

Example:

```python
PAPER_RESULTS = {
    "ML-KEM-512": {
        "AUROC": None,
        "Advantage": None
    }
}
```

Generate comparison tables.

Compute:

```python
absolute_difference
relative_difference
```

for each metric.

---

# Section 12 — Export Results

Save:

```text
phase_transition.csv
secret_diversity.csv
sample_efficiency.csv
```

and all JSON outputs.

Create:

```python
results_summary.xlsx
```

containing every experiment.

---

# Section 13 — Reproducibility Metadata

Automatically save:

```python
Python version
Torch version
CUDA version
Git commit hash
Date
GPU model
```

into:

```python
reproducibility.json
```

---

# Notebook Quality Requirements

The notebook must:

* Use subprocess.run()
* Check return codes
* Raise exceptions on failures
* Display progress bars where possible
* Save all outputs to disk
* Be restart-safe
* Skip completed experiments when outputs already exist
* Avoid hardcoded paths
* Work in Google Colab
* Work in VS Code Jupyter

Generate the notebook as a complete executable notebook structure with code cells and markdown cells.
