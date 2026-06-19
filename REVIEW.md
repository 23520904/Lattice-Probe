# Lattice-Probe Audit: Secret Handling Verification

I need a code-level audit of dataset generation and secret management.

This is NOT a request for opinions.

Inspect the actual implementation and determine exactly how secrets are generated, stored, and reused.

---

# Background

The CLI extraction indicates:

```bash
python scripts/generate_dataset.py \
  --param-set ML-KEM-512 \
  --output-dir data/train
```

creates:

```text
shard_*.npz
secret.npy
```

inside the output directory.

This raises a critical scientific concern:

If train, validation, and test datasets are generated independently, they may all contain different secrets.

Example:

```text
data/train/secret.npy
data/val/secret.npy
data/test/secret.npy
```

If these differ, then every reported result may already be measuring cross-secret generalization rather than same-secret distinguishing.

---

# Tasks

## Part 1 — Inspect Secret Generation

Trace the entire code path:

```text
scripts/generate_dataset.py
↓
src/latticeprobe/datasets.py
↓
src/latticeprobe/sampler.py
↓
any helper modules
```

Determine:

### How many secrets are generated?

For example:

```text
One secret per dataset
One secret per shard
One secret per sample
Multiple secrets per dataset
```

Provide exact code references.

---

## Part 2 — Train / Val / Test Semantics

Determine whether:

```text
train
val
test
```

generated via separate invocations of:

```bash
python scripts/generate_dataset.py ...
```

will automatically share the same secret.

Or whether they generate independent secrets.

Provide definitive evidence from the code.

---

## Part 3 — Secret Reuse Audit

Answer the following questions:

### Question A

Does:

```bash
python scripts/generate_dataset.py \
  --output-dir train
```

always create a fresh secret?

---

### Question B

Can the script reuse an existing secret?

For example:

```text
--secret-file
--reuse-secret
```

or equivalent.

---

### Question C

Can a user intentionally generate:

```text
Train Secret = A
Val Secret = A
Test Secret = A
```

without modifying source code?

If yes, provide commands.

If no, explain why.

---

## Part 4 — Current Experimental Regime

Based on the current implementation, determine which regime is actually being used:

### Regime 1

```text
Single Secret
Many Samples
```

### Regime 2

```text
Different Secret Per Dataset Split
```

### Regime 3

```text
Multiple Secrets Per Dataset
```

### Regime 4

```text
Fresh Secret Per Sample
```

Provide evidence.

---

## Part 5 — Scientific Consequences

If train/val/test currently use different secrets:

Explain:

```text
What conclusions remain valid
What conclusions change
Which paper tables are affected
```

Be precise.

Do not speculate.

---

## Part 6 — Recommended Fix

If the implementation is scientifically ambiguous:

Propose a minimal fix.

Requirements:

- Backward compatible
- Reproducible
- Explicit secret control
- Suitable for publication

Examples:

```text
--secret-file
--save-secret
--reuse-secret
```

or similar.

Provide:

### Proposed CLI

### Code Changes Required

### Benefits

### Risks

---

IMPORTANT

Do not assume intended behavior.

Report only what the source code actually does.