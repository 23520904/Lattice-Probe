# Lattice-Probe CLI and Experiment Interface Extraction

I need you to inspect the CURRENT codebase and produce a complete execution guide.

Do not summarize.

Do not describe architecture.

Do not provide opinions.

Instead, extract the actual runnable commands and argument definitions from the source code.

---

## Files To Inspect

Inspect all experiment-related files, including but not limited to:

```text
scripts/generate_dataset.py
scripts/train.py
scripts/evaluate.py
scripts/bit_recovery.py

src/latticeprobe/params.py
src/latticeprobe/datasets.py
src/latticeprobe/sampler.py
```

and any helper modules that define CLI arguments.

---

# Part 1 — CLI Reference

For every script provide:

## Script

```text
<filename>
```

## Usage

```bash
python <script> --help
```

## Arguments

For every argument report:

| Argument | Type | Default | Required | Description |
|----------|--------|----------|----------|----------|

Include ALL arguments.

Do not omit newly added flags.

Examples:

```text
--noise-scale
--shuffle-labels
--seed
--representation
--model
--param-set
--checkpoint
```

etc.

---

# Part 2 — Available Parameter Sets

Inspect params.py and list every parameter set.

Output:

| Name | n | k | q | Noise Distribution | Notes |
|--------|--------|--------|--------|--------|--------|

Include:

- ML-KEM-512
- ML-KEM-768
- ML-KEM-1024
- weakened variants
- edge variants
- custom variants

Use exact names expected by the CLI.

---

# Part 3 — Dataset Generation Commands

Provide actual commands for generating:

## Standard ML-KEM-512

```bash
...
```

## Standard ML-KEM-768

```bash
...
```

## Standard ML-KEM-1024

```bash
...
```

## Edge-of-Margin Dataset

```bash
...
```

## Weakened Dataset

```bash
...
```

Use exact argument names from the code.

---

# Part 4 — Training Commands

Provide complete runnable commands for:

## Transformer

```bash
...
```

## GNN

```bash
...
```

For each command include:

- batch size
- learning rate
- epochs
- scheduler settings
- checkpoint directory

using the current defaults from the code.

---

# Part 5 — Evaluation Commands

Provide exact commands for:

## Normal Evaluation

```bash
...
```

## Permutation Test

```bash
...
```

## Cross-Secret Evaluation

```bash
...
```

## Bit Recovery

```bash
...
```

---

# Part 6 — Reproduce Paper Tables

Inspect the codebase and identify which parameter combinations correspond to the tables reported in the paper.

For every table provide:

## Table Name

### Dataset Generation

```bash
...
```

### Training

```bash
...
```

### Evaluation

```bash
...
```

### Expected Output Metrics

- AUROC
- Advantage
- BCa Confidence Interval
- Cohen's d
- Logit Separation

---

# Part 7 — New Research Experiments

Provide exact commands for:

## Noise Phase Transition

Generate commands for:

```text
1.00
0.95
0.90
0.85
0.80
0.75
0.70
0.65
0.60
0.55
0.50
```

noise scales.

---

## Cross-Secret Transfer

Train:

```text
Secret A
```

Test:

```text
Secret B
```

Provide exact commands.

---

## Secret Diversity Scaling

If supported by the codebase, provide commands for:

```text
1 secret
10 secrets
100 secrets
1000 secrets
```

Otherwise explain what code changes are required.

---

## Sample Complexity Scaling

Provide commands for:

```text
2^10
2^12
2^14
2^16
2^18
2^20
2^22
```

samples.

---

# Part 8 — Colab Execution Plan

Produce a recommended execution order.

Rank experiments by:

| Experiment | Estimated Runtime | GPU Required | Priority |
|------------|------------|------------|------------|

Use realistic estimates based on the current code.

---

IMPORTANT

Use ONLY information present in the codebase.

Do not invent commands.

Do not assume flag names.

Do not guess defaults.

If a command cannot be determined from source code, explicitly state:

```text
Not found in codebase
```