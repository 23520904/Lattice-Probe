"""
End-to-end pipeline tests: generate → train (1 epoch) → evaluate.

Designed to run on CPU in < 90 s without any GPU or large data.
Each test uses pytest's tmp_path fixture for full isolation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Make src/ importable when running from repo root without editable install.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from latticeprobe.params import get_params


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def params_512():
    return get_params("ML-KEM-512")


def _make_args(overrides: dict) -> SimpleNamespace:
    """Build a SimpleNamespace that mimics the argparse Namespace from train.py.

    Uses tiny model sizes (d_model=64, 2 heads, 2 layers) so pipeline tests
    run quickly on CPU without risking OOM with the full 512-dim architecture.
    """
    defaults = dict(
        param_set="ML-KEM-512",
        model="transformer",
        train_dir=None,
        val_dir=None,
        output_dir=None,
        epochs=1,
        batch_size=4,          # small to avoid OOM on CPU
        lr=1e-4,
        weight_decay=1e-2,
        patience=5,
        ckpt_every=1,
        device="cpu",
        wandb=False,
        wandb_project="latticeprobe-test",
        compute_log=None,
        # Tiny model sizes for fast tests
        d_model=64,
        nhead=4,
        num_layers=2,
        ff_dim=128,
        hidden=32,
        gnn_layers=2,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Phase 5 — Dataset generation ──────────────────────────────────────────────

class TestGenerateDataset:
    def test_shards_created(self, tmp_path):
        from generate_dataset import generate_dataset
        out = generate_dataset(
            param_set="ML-KEM-512",
            n_samples=64,
            output_dir=str(tmp_path / "data"),
            shard_size=32,
            quiet=True,
        )
        shards = sorted(out.glob("shard_*.npz"))
        assert len(shards) == 2, f"Expected 2 shards, got {len(shards)}"

    def test_shard_arrays_shape(self, tmp_path):
        from generate_dataset import generate_dataset
        out = generate_dataset(
            param_set="ML-KEM-512",
            n_samples=64,
            output_dir=str(tmp_path / "data"),
            shard_size=64,
            quiet=True,
        )
        d = np.load(out / "shard_00000.npz")
        assert d["a"].shape     == (64, 2, 256)
        assert d["b"].shape     == (64, 256)
        assert d["label"].shape == (64,)

    def test_labels_binary(self, tmp_path):
        from generate_dataset import generate_dataset
        out = generate_dataset(
            param_set="ML-KEM-512",
            n_samples=128,
            output_dir=str(tmp_path / "data"),
            shard_size=128,
            quiet=True,
        )
        d = np.load(out / "shard_00000.npz")
        assert set(d["label"].tolist()).issubset({0, 1})

    def test_secret_saved(self, tmp_path):
        from generate_dataset import generate_dataset
        out = generate_dataset(
            param_set="ML-KEM-512",
            n_samples=32,
            output_dir=str(tmp_path / "data"),
            shard_size=32,
            quiet=True,
        )
        secret = np.load(out / "secret.npy")
        assert secret.shape == (2, 256)  # k=2, n=256

    def test_w3_gaussian_noise(self, tmp_path):
        """W3 uses discrete Gaussian noise — generation must not crash."""
        from generate_dataset import generate_dataset
        out = generate_dataset(
            param_set="W3",
            n_samples=32,
            output_dir=str(tmp_path / "w3"),
            shard_size=32,
            quiet=True,
        )
        assert len(list(out.glob("shard_*.npz"))) == 1

    def test_different_param_sets(self, tmp_path):
        from generate_dataset import generate_dataset
        for ps in ["ML-KEM-768", "W1", "W2", "edge"]:
            out = generate_dataset(
                param_set=ps,
                n_samples=32,
                output_dir=str(tmp_path / ps),
                shard_size=32,
                quiet=True,
            )
            assert len(list(out.glob("shard_*.npz"))) >= 1


# ── Phase 5 — Training pipeline ───────────────────────────────────────────────

def _generate_split(tmp_path: Path, param_set: str, n: int, name: str) -> str:
    from generate_dataset import generate_dataset
    out = generate_dataset(
        param_set=param_set,
        n_samples=n,
        output_dir=str(tmp_path / name),
        shard_size=n,
        quiet=True,
    )
    return str(out)


class TestTraining:
    def test_transformer_runs_one_epoch(self, tmp_path):
        from train import train
        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "train")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "val")
        ckpt_log  = str(tmp_path / "compute_log.csv")

        args = _make_args(dict(
            model="transformer",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "ckpt_t"),
            compute_log=ckpt_log,
        ))
        best = train(args)
        assert Path(best).exists(), "best.pt was not created"

    def test_gnn_runs_one_epoch(self, tmp_path):
        from train import train
        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "train_g")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "val_g")
        ckpt_log  = str(tmp_path / "compute_log.csv")

        args = _make_args(dict(
            model="gnn",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "ckpt_g"),
            compute_log=ckpt_log,
        ))
        best = train(args)
        assert Path(best).exists(), "best.pt was not created"

    def test_checkpoint_contains_required_keys(self, tmp_path):
        from train import train
        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "train_k")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "val_k")

        args = _make_args(dict(
            model="transformer",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "ckpt_k"),
            compute_log=str(tmp_path / "log.csv"),
        ))
        best = train(args)
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        for key in ("epoch", "val_auroc", "model_state", "args"):
            assert key in ckpt, f"Missing key '{key}' in checkpoint"

    def test_compute_log_written(self, tmp_path):
        from train import train
        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "train_l")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "val_l")
        log_path  = str(tmp_path / "compute_log.csv")

        args = _make_args(dict(
            model="transformer",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "ckpt_l"),
            compute_log=log_path,
        ))
        train(args)
        assert Path(log_path).exists(), "compute_log.csv not created"
        content = Path(log_path).read_text()
        assert "val_auroc" in content
        assert "ML-KEM-512" in content

    def test_periodic_checkpoint_saved(self, tmp_path):
        """With ckpt_every=1 and epochs=2, two periodic checkpoints should exist."""
        from train import train
        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "train_p")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "val_p")

        args = _make_args(dict(
            model="transformer",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "ckpt_p"),
            compute_log=str(tmp_path / "log_p.csv"),
            epochs=2,
            ckpt_every=1,
        ))
        train(args)
        periodic = list(Path(args.output_dir).glob("ckpt_epoch*.pt"))
        assert len(periodic) == 2, f"Expected 2 periodic checkpoints, got {len(periodic)}"

    def test_early_stopping(self, tmp_path):
        """With patience=1 and random weights, training should stop after 2 epochs at most."""
        from train import train
        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "train_es")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "val_es")
        log_path  = str(tmp_path / "log_es.csv")

        args = _make_args(dict(
            model="transformer",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "ckpt_es"),
            compute_log=log_path,
            epochs=20,
            patience=1,
        ))
        train(args)
        import csv
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        # Should have stopped early — max 2 epochs (1 best + 1 patience step)
        assert len(rows) <= 3, f"Early stopping didn't trigger: {len(rows)} epochs"


# ── Phase 6 — Evaluation ──────────────────────────────────────────────────────

class TestEvaluation:
    def test_evaluate_model_returns_auroc(self, tmp_path):
        """Train for 1 epoch then evaluate — verify AUROC is in [0, 1]."""
        from evaluate import evaluate_model
        from train import train

        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "ev_train")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "ev_val")
        test_dir  = _generate_split(tmp_path, "ML-KEM-512", 32, "ev_test")

        args = _make_args(dict(
            model="transformer",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "ev_ckpt"),
            compute_log=str(tmp_path / "ev_log.csv"),
        ))
        best = train(args)

        results = evaluate_model(
            checkpoint=str(best),
            model_name="transformer",
            param_set="ML-KEM-512",
            test_dir=test_dir,
            train_dir=train_dir,
            batch_size=32,
            n_boot=10,
            device_arg="cpu",
        )
        assert 0.0 <= results["model_auroc"] <= 1.0
        assert "model_auroc_ci_lo" in results
        assert "model_auroc_ci_hi" in results

    def test_evaluate_gnn(self, tmp_path):
        from evaluate import evaluate_model
        from train import train

        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "gev_train")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "gev_val")
        test_dir  = _generate_split(tmp_path, "ML-KEM-512", 32, "gev_test")

        args = _make_args(dict(
            model="gnn",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "gev_ckpt"),
            compute_log=str(tmp_path / "gev_log.csv"),
        ))
        best = train(args)

        results = evaluate_model(
            checkpoint=str(best),
            model_name="gnn",
            param_set="ML-KEM-512",
            test_dir=test_dir,
            batch_size=32,
            n_boot=10,
            device_arg="cpu",
        )
        assert 0.0 <= results["model_auroc"] <= 1.0

    def test_evaluate_outputs_json(self, tmp_path):
        from evaluate import evaluate_model
        from train import train

        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "jt")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "jv")
        test_dir  = _generate_split(tmp_path, "ML-KEM-512", 32, "jte")

        args = _make_args(dict(
            model="transformer",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "j_ckpt"),
            compute_log=str(tmp_path / "j_log.csv"),
        ))
        best = train(args)
        out_json = str(tmp_path / "result.json")

        results = evaluate_model(
            checkpoint=str(best),
            model_name="transformer",
            param_set="ML-KEM-512",
            test_dir=test_dir,
            batch_size=32,
            n_boot=5,
            device_arg="cpu",
        )
        Path(out_json).write_text(json.dumps(results, indent=2))
        loaded = json.loads(Path(out_json).read_text())
        assert loaded["param_set"] == "ML-KEM-512"
        assert loaded["model"] == "transformer"

    def test_chi2_baseline_present(self, tmp_path):
        from evaluate import evaluate_model
        from train import train

        train_dir = _generate_split(tmp_path, "ML-KEM-512", 64, "chi_t")
        val_dir   = _generate_split(tmp_path, "ML-KEM-512", 32, "chi_v")
        test_dir  = _generate_split(tmp_path, "ML-KEM-512", 64, "chi_te")

        args = _make_args(dict(
            model="transformer",
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=str(tmp_path / "chi_ckpt"),
            compute_log=str(tmp_path / "chi_log.csv"),
        ))
        best = train(args)
        results = evaluate_model(
            checkpoint=str(best),
            model_name="transformer",
            param_set="ML-KEM-512",
            test_dir=test_dir,
            batch_size=32,
            n_boot=5,
            device_arg="cpu",
        )
        assert "chi2_auroc" in results


# ── Phase 6 — Partial-bit recovery ───────────────────────────────────────────

class TestBitRecovery:
    def test_bit_recovery_returns_per_bit_accuracy(self, tmp_path):
        from bit_recovery import recover_bits

        train_dir = _generate_split(tmp_path, "ML-KEM-512", 128, "br_train")
        test_dir  = _generate_split(tmp_path, "ML-KEM-512", 64,  "br_test")

        results = recover_bits(
            train_dir=train_dir,
            test_dir=test_dir,
            param_set="ML-KEM-512",
            bit_range="0:8",   # test only 8 bits for speed
        )
        assert "mean_accuracy" in results
        assert results["n_bits_tested"] == 8
        assert isinstance(results["per_bit_accuracy"], list)
        assert len(results["per_bit_accuracy"]) == 8

    def test_bit_accuracy_in_valid_range(self, tmp_path):
        from bit_recovery import recover_bits

        train_dir = _generate_split(tmp_path, "ML-KEM-512", 128, "br2_train")
        test_dir  = _generate_split(tmp_path, "ML-KEM-512", 64,  "br2_test")

        results = recover_bits(
            train_dir=train_dir,
            test_dir=test_dir,
            param_set="ML-KEM-512",
            bit_range="0:4",
        )
        # All valid per-bit accuracies must be in [0, 1]
        valid = [a for a in results["per_bit_accuracy"] if not np.isnan(a)]
        for acc in valid:
            assert 0.0 <= acc <= 1.0, f"Invalid accuracy: {acc}"

    def test_missing_secret_returns_error(self, tmp_path):
        """If secret.npy is not present, recover_bits returns an error dict."""
        from bit_recovery import recover_bits

        # Generate a directory without secret (can't directly disable saving,
        # so we delete it after generation)
        train_dir = _generate_split(tmp_path, "ML-KEM-512", 32, "ns_train")
        test_dir  = _generate_split(tmp_path, "ML-KEM-512", 32, "ns_test")
        (Path(train_dir) / "secret.npy").unlink()
        (Path(test_dir)  / "secret.npy").unlink()

        results = recover_bits(
            train_dir=train_dir,
            test_dir=test_dir,
            param_set="ML-KEM-512",
            bit_range="0:2",
        )
        assert "error" in results
