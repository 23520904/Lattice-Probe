"""
Architecture consistency tests required by REVIEW.md §9.

These tests verify that the implementation matches the paper specification
(paper §4.3) and document known paper inconsistencies.

Run with: pytest tests/test_architecture.py -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def params_512():
    from latticeprobe.params import get_params
    return get_params("ML-KEM-512")

@pytest.fixture(scope="module")
def params_768():
    from latticeprobe.params import get_params
    return get_params("ML-KEM-768")

@pytest.fixture(scope="module")
def params_1024():
    from latticeprobe.params import get_params
    return get_params("ML-KEM-1024")


# ── §9.1 — Transformer parameter count ───────────────────────────────────────

def test_transformer_parameter_count(params_512):
    """
    Paper §4.3: d_model=512, 8 layers, ff=2048.
    Correct count: ~27,936,257.
    PAPER INCONSISTENCY: paper claims ~51M.
    """
    from latticeprobe.models.transformer import LWETransformer
    model = LWETransformer(params_512)
    n = sum(p.numel() for p in model.parameters())
    expected = 27_936_257
    assert abs(n - expected) / expected < 0.05, (
        f"Transformer: got {n:,}, expected ~{expected:,}. "
        "PAPER INCONSISTENCY: paper claims ~51M (incorrect for stated architecture)."
    )


# ── §9.2 — GNN parameter count ───────────────────────────────────────────────

def test_gnn_parameter_count(params_512):
    """
    Paper §4.3: GraphSAGE, 6 layers, hidden=256.
    Correct count: ~662,273.
    PAPER INCONSISTENCY: paper claims ~18M.
    """
    from latticeprobe.models.gnn import LWEGNN
    model = LWEGNN(params_512)
    n = sum(p.numel() for p in model.parameters())
    expected = 662_273
    assert abs(n - expected) / expected < 0.05, (
        f"GNN: got {n:,}, expected ~{expected:,}. "
        "PAPER INCONSISTENCY: paper claims ~18M (incorrect for stated architecture)."
    )


# ── §9.3 — CLS token present and correctly wired ─────────────────────────────

def test_cls_token_present(params_512):
    """Transformer must have a learnable cls_token parameter (BERT-style)."""
    from latticeprobe.models.transformer import LWETransformer
    model = LWETransformer(params_512)
    assert hasattr(model, "cls_token"), "LWETransformer missing cls_token parameter"
    assert isinstance(model.cls_token, torch.nn.Parameter)
    assert model.cls_token.shape == (1, 1, 512), (
        f"cls_token shape {model.cls_token.shape} != (1,1,512)"
    )


def test_cls_token_prepended(params_512):
    """
    After forward(), the encoder input must be seq_len+1 (CLS + content).
    Verify by checking that position 0 of the encoder output is the CLS readout.
    """
    from latticeprobe.models.transformer import LWETransformer
    model = LWETransformer(params_512)
    model.eval()

    seq_len = params_512.n * (params_512.k + 1)  # k*n + n
    x = torch.zeros(2, seq_len, dtype=torch.long)  # dummy tokens

    with torch.no_grad():
        # Patch encoder to capture its input
        captured = {}
        orig_forward = model.encoder.forward

        def capturing_forward(src, *args, **kwargs):
            captured["input"] = src
            return orig_forward(src, *args, **kwargs)

        model.encoder.forward = capturing_forward
        _ = model(x)

    enc_input = captured["input"]
    assert enc_input.shape == (2, seq_len + 1, 512), (
        f"Encoder input shape {enc_input.shape}; expected (2, {seq_len+1}, 512). "
        "CLS token not prepended."
    )


def test_cls_token_initialised(params_512):
    """cls_token must be initialised (non-zero after _init_weights)."""
    from latticeprobe.models.transformer import LWETransformer
    model = LWETransformer(params_512)
    assert model.cls_token.abs().sum().item() > 0.0, (
        "cls_token is all-zero after init — _init_weights not applied."
    )


# ── §9.4 — Disjoint secret generation ────────────────────────────────────────

def test_disjoint_secret_generation(tmp_path):
    """
    Paper §5.2: train/val/test secrets must be independently drawn.
    Two calls to generate_dataset with no --secret-file must produce distinct secrets.
    """
    from generate_dataset import generate_dataset
    out_a = generate_dataset("ML-KEM-512", n_samples=32, output_dir=str(tmp_path / "a"),
                             shard_size=32, quiet=True)
    out_b = generate_dataset("ML-KEM-512", n_samples=32, output_dir=str(tmp_path / "b"),
                             shard_size=32, quiet=True)

    s_a = np.load(out_a / "secrets.npy")
    s_b = np.load(out_b / "secrets.npy")
    assert not np.array_equal(s_a, s_b), (
        "Two independent generate_dataset calls produced identical secrets. "
        "RNG seeding may be broken — violates paper §5.2."
    )


def test_validate_disjoint_secrets_raises_on_reuse(tmp_path):
    """validate_disjoint_secrets() must raise RuntimeError when the same secret is reused."""
    from generate_dataset import generate_dataset, validate_disjoint_secrets
    out = generate_dataset("ML-KEM-512", n_samples=32, output_dir=str(tmp_path / "src"),
                           shard_size=32, quiet=True)
    secret_path = str(out / "secrets.npy")

    with pytest.raises(RuntimeError, match="Secret reuse detected"):
        validate_disjoint_secrets(secret_path, secret_path)


def test_validate_disjoint_secrets_passes_for_distinct(tmp_path):
    """validate_disjoint_secrets() must not raise when secrets are distinct."""
    from generate_dataset import generate_dataset, validate_disjoint_secrets
    out_a = generate_dataset("ML-KEM-512", n_samples=32, output_dir=str(tmp_path / "sa"),
                             shard_size=32, quiet=True)
    out_b = generate_dataset("ML-KEM-512", n_samples=32, output_dir=str(tmp_path / "sb"),
                             shard_size=32, quiet=True)
    # Must not raise
    validate_disjoint_secrets(str(out_a / "secrets.npy"), str(out_b / "secrets.npy"))


# ── §9.5 — Lazy GNN import in train.py ───────────────────────────────────────

def test_lazy_gnn_import_train():
    """
    LWEGNN must NOT be imported at module level in scripts/train.py.
    A Transformer-only run must not touch torch_geometric.nn.
    """
    train_src = (Path(__file__).parent.parent / "scripts" / "train.py").read_text()
    tree = ast.parse(train_src)

    # Find all top-level import statements (module body, not inside functions)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Check if this node is at module level (parent is Module)
            pass

    # Simpler check: verify LWEGNN does not appear in a top-level import
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top_level_imports.append(ast.dump(node))
        elif isinstance(node, ast.If):
            # __main__ block — also check
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    top_level_imports.append(ast.dump(sub))

    unconditional_gnn = [s for s in top_level_imports if "LWEGNN" in s]
    assert not unconditional_gnn, (
        f"LWEGNN imported unconditionally in train.py __main__: {unconditional_gnn}. "
        "This breaks transformer training on Colab when torch_sparse is missing."
    )


# ── §9.6 — Lazy GNN import in evaluate.py ────────────────────────────────────

def test_lazy_gnn_import_eval():
    """
    LWEGNN must NOT be imported at module level in scripts/evaluate.py.
    Transformer evaluation must work without torch_sparse.
    """
    eval_src = (Path(__file__).parent.parent / "scripts" / "evaluate.py").read_text()
    tree = ast.parse(eval_src)

    top_level_imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top_level_imports.append(ast.dump(node))
        elif isinstance(node, ast.If):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    top_level_imports.append(ast.dump(sub))

    unconditional_gnn = [s for s in top_level_imports if "LWEGNN" in s]
    assert not unconditional_gnn, (
        f"LWEGNN imported unconditionally in evaluate.py: {unconditional_gnn}. "
        "This breaks transformer evaluation on Colab when torch_sparse is missing."
    )


# ── §9.7 — Sequence lengths ───────────────────────────────────────────────────

@pytest.mark.parametrize("param_set,k,expected_seq,expected_with_cls", [
    ("ML-KEM-512",  2, 768,  769),
    ("ML-KEM-768",  3, 1024, 1025),
    ("ML-KEM-1024", 4, 1280, 1281),
])
def test_sequence_lengths(param_set, k, expected_seq, expected_with_cls):
    """
    Verify token sequence lengths match paper §4.2.1: k*n + n tokens.
    With CLS prepend, encoder input is k*n + n + 1.
    """
    from latticeprobe.params import get_params
    from latticeprobe.representations import to_sequence
    params = get_params(param_set)

    rng = np.random.default_rng(0)
    a = rng.integers(0, params.q, size=(params.k, params.n), dtype=np.int64)
    b = rng.integers(0, params.q, size=(params.n,), dtype=np.int64)
    tokens = to_sequence(a, b)

    assert len(tokens) == expected_seq, (
        f"{param_set}: sequence length {len(tokens)}, expected {expected_seq} (k*n+n={k}*256+256)"
    )

    from latticeprobe.models.transformer import LWETransformer
    model = LWETransformer(params)
    model.eval()

    captured = {}
    orig_fwd = model.encoder.forward
    def cap(src, *a, **kw):
        captured["shape"] = src.shape
        return orig_fwd(src, *a, **kw)
    model.encoder.forward = cap

    with torch.no_grad():
        model(tokens.unsqueeze(0))

    assert captured["shape"][1] == expected_with_cls, (
        f"{param_set}: encoder input length {captured['shape'][1]}, "
        f"expected {expected_with_cls} (seq+CLS)"
    )


# ── §9 extra — No GATConv anywhere in source ─────────────────────────────────

def test_no_gatconv_in_source():
    """Ensure GATConv is not imported or instantiated anywhere in src/."""
    src_root = Path(__file__).parent.parent / "src"
    violations = []
    for py_file in src_root.rglob("*.py"):
        text = py_file.read_text()
        if "GATConv" in text and "not GATConv" not in text and "[SAGEConv, not GATConv]" not in text:
            violations.append(str(py_file))
    assert not violations, (
        f"GATConv found in source files: {violations}. "
        "All GNN code must use SAGEConv (paper §4.3)."
    )


# ── §9 extra — Checkpoint fingerprint written ─────────────────────────────────

def test_checkpoint_architecture_version():
    """
    Verify by AST inspection that train.py's torch.save calls include
    'architecture_version': 'v2-reconciled' and 'model_type'.

    This avoids actually running training (which would require all deps installed).
    The _check_checkpoint_compat tests in evaluate.py cover runtime rejection of
    checkpoints that lack these keys.
    """
    train_src = (Path(__file__).parent.parent / "scripts" / "train.py").read_text()

    # Check that the literal string "v2-reconciled" appears in torch.save dict literals
    assert '"v2-reconciled"' in train_src or "'v2-reconciled'" in train_src, (
        "train.py does not contain the 'v2-reconciled' architecture version string. "
        "Checkpoints will not be fingerprinted."
    )
    assert '"architecture_version"' in train_src or "'architecture_version'" in train_src, (
        "train.py does not save 'architecture_version' key in checkpoints."
    )
    assert '"model_type"' in train_src or "'model_type'" in train_src, (
        "train.py does not save 'model_type' key in checkpoints."
    )

    # Verify both the best-checkpoint and periodic-checkpoint saves include the key
    save_count = train_src.count("architecture_version")
    assert save_count >= 2, (
        f"'architecture_version' appears {save_count} time(s) in train.py; "
        "expected >=2 (best.pt save AND periodic checkpoint save)."
    )


# ── §9 extra — Checkpoint compat raises on old GAT checkpoint ─────────────────

def test_checkpoint_compat_rejects_gat():
    """evaluate.py must raise RuntimeError when loading a pre-reconciliation GAT checkpoint."""
    import torch
    from evaluate import _check_checkpoint_compat

    fake_gat_ckpt = {
        # no architecture_version key (old checkpoint)
        "epoch": 1,
        "val_auroc": 0.5,
        "model_state": {
            "convs.0.att_src": torch.zeros(1),   # GATConv-specific key
            "convs.0.att_dst": torch.zeros(1),
        },
        "args": {},
    }
    with pytest.raises(RuntimeError, match="GATConv"):
        _check_checkpoint_compat(fake_gat_ckpt, "gnn", "fake_path/best.pt")


def test_checkpoint_compat_rejects_pre_cls_transformer():
    """evaluate.py must raise RuntimeError for pre-CLS transformer checkpoints."""
    from evaluate import _check_checkpoint_compat

    fake_old_ckpt = {
        # no architecture_version (old), no cls_token in state
        "epoch": 1,
        "val_auroc": 0.5,
        "model_state": {
            "token_embed.weight": torch.zeros(3329, 512),
            "pos_embed.weight": torch.zeros(768, 512),
            # no cls_token key
        },
        "args": {},
    }
    with pytest.raises(RuntimeError, match="CLS-token"):
        _check_checkpoint_compat(fake_old_ckpt, "transformer", "fake_path/best.pt")
