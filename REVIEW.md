You are acting as a senior cryptography research engineer and reproducibility auditor.

Project: LatticeProbe

Your task is to perform a FULL IMPLEMENTATION AUDIT against the specification below and modify the codebase so that it is internally consistent, paper-consistent wherever possible, and fully reproducible.

IMPORTANT:

* The specification below is the source of truth.
* Do NOT blindly follow the paper when the paper is mathematically inconsistent.
* When the paper is self-contradictory, preserve the mathematically valid implementation and document the inconsistency.
* Produce code changes, not commentary.
* Update docstrings, comments, notebooks, checkpoints metadata, and configuration defaults where required.
* At the end generate a REPORT.md summarizing every modification and whether it was:

  * FIXED
  * DOCUMENTED
  * PAPER INCONSISTENCY
  * OPEN ISSUE

========================================
AUDIT REQUIREMENTS
==================

1. VERIFY GNN IMPLEMENTATION

Expected architecture:

* GraphSAGE backbone
* SAGEConv
* hidden_dim=256
* num_layers=6
* BatchNorm after each layer
* ReLU
* Dropout(0.1)
* global_mean_pool
* Linear(hidden,1)

Required actions:

* Ensure NO GATConv remains anywhere.

* Ensure no attention-specific logic remains.

* Ensure edge_attr is not passed into SAGEConv.

* Remove dead code related to GAT.

* Verify forward() uses:

  x = conv(x, edge_index)

* Add parameter count assertion utility.

Expected parameter count:

~662,273 parameters

Document paper claim of 18M as inconsistent.

========================================

2. VERIFY TRANSFORMER IMPLEMENTATION

Expected architecture:

* d_model=512
* nhead=8
* num_layers=8
* dim_feedforward=2048
* dropout=0.1
* norm_first=True
* batch_first=True

Required actions:

* Verify learnable CLS token exists.
* Verify CLS token is prepended.
* Verify h[:,0] corresponds to CLS output.
* Verify position embeddings correctly handle CLS prepend.
* Verify initialization includes cls_token.

Expected parameter count:

~27.9M

Add parameter counting utility and automated check.

If code or configs still mention:

* nhead=12
* ~51M parameters

replace with explanatory note:

"PAPER INCONSISTENCY: d_model=512 is not divisible by 12. Implementation uses nhead=8."

========================================

3. DATASET GENERATION VALIDATION

Paper requirement:

train secret != val secret != test secret

Required actions:

* Search entire repository for:
  --secret-file
  secrets.npy

* Verify validation and test generation never reuse train secret.

* Add runtime validation:

  assert train_secret_hash != val_secret_hash
  assert train_secret_hash != test_secret_hash

where applicable.

* If secret reuse is detected:
  raise RuntimeError

========================================

4. TRAINING PIPELINE VALIDATION

Verify:

scripts/train.py

Requirements:

* Transformer path must not require torch_geometric.
* LWEGNN import must be lazy.
* build_model() must import LWEGNN only inside GNN branch.

If any unconditional import exists:

```
from latticeprobe.models.gnn import LWEGNN
```

remove it.

========================================

5. EVALUATION PIPELINE VALIDATION

Verify:

scripts/evaluate.py

Requirements:

* Transformer evaluation must work without torch_sparse.
* LWEGNN import must be lazy.

Move all GNN imports inside GNN-only code paths.

========================================

6. NOTEBOOK REPRODUCIBILITY

Audit:

notebooks/
notebook/

Required:

* train_dir
* val_dir
* test_dir

must be defined before preflight checks.

Training cells must explicitly use:

```
--batch-size 32
```

when running transformer on T4.

Add explanatory comments documenting memory calculations.

========================================

7. CHECKPOINT COMPATIBILITY SAFETY

Add architecture fingerprinting.

Checkpoint should store:

{
"architecture_version": "...",
"model_type": "...",
"args": ...
}

During loading:

* Reject incompatible checkpoints.
* Detect old GAT checkpoints.
* Detect pre-CLS transformer checkpoints.

Raise informative error:

"Checkpoint was trained before architecture reconciliation and must be retrained."

========================================

8. DOCUMENTATION RECONCILIATION

Update:

README.md
docs/*
notebooks markdown cells

Ensure documentation states:

Transformer:
~27.9M parameters

GNN:
~0.66M parameters

Add section:

"Paper inconsistencies"

including:

* nhead=12 impossible for d_model=512
* transformer 51M claim inconsistent
* GNN 18M claim inconsistent
* CLS pooling unspecified
* FFN dimension unspecified

========================================

9. AUTOMATED CONSISTENCY TESTS

Create tests that verify:

test_transformer_parameter_count()

test_gnn_parameter_count()

test_cls_token_present()

test_disjoint_secret_generation()

test_lazy_gnn_import_train()

test_lazy_gnn_import_eval()

test_sequence_lengths()

Expected sequence lengths:

ML-KEM-512:
768 → 769 with CLS

ML-KEM-768:
1024 → 1025

ML-KEM-1024:
1280 → 1281

========================================

10. OPEN ISSUES TO FLAG

Do NOT attempt speculative fixes.

Leave TODO markers for:

* Sections 8–10 of Lattice_Probe_Experiments.ipynb if they remain placeholders.
* Any paper omission that cannot be reconstructed from the PDF.

Mark them as:

OPEN ISSUE

========================================

DELIVERABLES

Return:

1. Unified diff of every modified file.
2. REPORT.md
3. List of checkpoints that must be retrained.
4. Results of all consistency tests.
5. Final statement indicating whether repository is:

* PAPER CONSISTENT
* IMPLEMENTATION CONSISTENT
* REPRODUCIBLE

with justification.
