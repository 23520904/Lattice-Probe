#!/usr/bin/env python3
"""
Evaluation harness for LatticeProbe models (paper §6).

Loads a trained checkpoint, runs inference on a test set, and reports:
  - Model AUROC with 95% bootstrap confidence interval (100 resamples)
  - χ² statistical distinguisher AUROC
  - Logistic regression and MLP baselines (requires --train-dir)
  - Cross-parameter table if --cross-param is set

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/transformer-512/best.pt \\
        --model transformer \\
        --param-set ML-KEM-512 \\
        --test-dir  data/ML-KEM-512/test \\
        [--train-dir data/ML-KEM-512/train] \\
        [--output-json results/transformer-512.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latticeprobe.params import PARAMS, get_params


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a trained LatticeProbe model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to a saved checkpoint (best.pt)")
    p.add_argument("--model",      required=True, choices=["transformer", "gnn"])
    p.add_argument("--param-set",  required=True, choices=list(PARAMS))
    p.add_argument("--test-dir",   required=True,
                   help="Directory containing test shard_*.npz files")
    p.add_argument("--train-dir",  default=None,
                   help="Training shard dir (needed for LR / MLP baselines)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-boot",     type=int, default=100,
                   help="Number of bootstrap resamples for AUROC CI")
    p.add_argument("--device",     default="auto")
    p.add_argument("--output-json", default=None,
                   help="Write full results to a JSON file")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="Permute labels to verify artifact immunity")
    p.add_argument("--repr", default="coeff", choices=["coeff", "ntt", "dual"],
                   help="Representation domain of the polynomials")
    return p.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def load_checkpoint(path: str, model_name: str, params, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    if model_name == "transformer":
        model = LWETransformer(
            params,
            d_model=saved_args.get("d_model", 512),
            nhead=saved_args.get("nhead", 8),
            num_layers=saved_args.get("num_layers", 8),
            dim_feedforward=saved_args.get("ff_dim", 2048),
        )
    else:
        model = LWEGNN(
            params,
            hidden=saved_args.get("hidden", 256),
            num_layers=saved_args.get("gnn_layers", 6),
        )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt.get("epoch"), ckpt.get("val_auroc")


def _build_loader(model_name: str, data_dir: str, params, batch_size: int, repr_type: str = "coeff"):
    if model_name == "transformer":
        ds = LWESequenceDataset(data_dir, params, repr_type=repr_type)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), ds
    else:
        from torch_geometric.loader import DataLoader as GeoLoader
        ds = LWEGraphDataset(data_dir, params, repr_type=repr_type)
        return GeoLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), ds


def _inference(model: torch.nn.Module, loader, device: torch.device,
               model_name: str) -> tuple[np.ndarray, np.ndarray]:
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if model_name == "transformer":
                tokens, labels = batch
                tokens = tokens.to(device)
                logits = model(tokens)
            else:
                data, labels = batch
                data   = data.to(device)
                logits = model(data)
            all_logits.append(logits.cpu().float().squeeze(1).numpy())
            all_labels.append(labels.float().reshape(-1).numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def _print_metrics(label: str, mean: float, lo: float, hi: float) -> None:
    adv = 2 * mean - 1
    adv_lo = 2 * lo - 1
    adv_hi = 2 * hi - 1
    print(f"  {label:<25s} AUROC = {mean:.4f} | Advantage = {adv:.4f} CI [{adv_lo:.4f}, {adv_hi:.4f}]")


# ── Evaluation functions ──────────────────────────────────────────────────────

def evaluate_model(
    checkpoint: str,
    model_name: str,
    param_set: str,
    test_dir: str,
    train_dir: Optional[str] = None,
    batch_size: int = 256,
    n_boot: int = 100,
    device_arg: str = "auto",
    shuffle_labels: bool = False,
    repr_type: str = "coeff",
) -> dict:
    """
    Full evaluation run. Returns a results dict.
    """
    params = get_params(param_set)
    device = resolve_device(device_arg)

    # Load model
    model, ckpt_epoch, ckpt_val_auroc = load_checkpoint(checkpoint, model_name, params, device)
    print(f"\nCheckpoint : {checkpoint}")
    print(f"  epoch={ckpt_epoch}  ckpt_val_auroc={ckpt_val_auroc}")

    # Inference
    test_loader, test_ds = _build_loader(model_name, test_dir, params, batch_size, repr_type=repr_type)
    logits, labels = _inference(model, test_loader, device, model_name)

    if shuffle_labels:
        print("  [EVAL] Shuffling labels for permutation testing...")
        np.random.shuffle(labels)

    # Logit Separation and Cohen's d
    lwe_logits = logits[labels == 1]
    unif_logits = logits[labels == 0]
    logit_sep = float(np.mean(lwe_logits) - np.mean(unif_logits)) if len(lwe_logits) and len(unif_logits) else 0.0
    
    n1, n2 = len(lwe_logits), len(unif_logits)
    v1 = np.var(lwe_logits, ddof=1) if n1 > 1 else 0.0
    v2 = np.var(unif_logits, ddof=1) if n2 > 1 else 0.0
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if n1 + n2 > 2 else 1e-9
    cohens_d = logit_sep / pooled_std

    mean, lo, hi = bootstrap_auroc(labels, logits, n_boot=n_boot)
    adv = 2 * mean - 1
    results: dict = {
        "param_set": param_set,
        "model": model_name,
        "checkpoint": checkpoint,
        "n_test": int(len(labels)),
        "model_auroc": float(mean),
        "model_advantage": float(adv),
        "model_advantage_ci_lo": float(2 * lo - 1),
        "model_advantage_ci_hi": float(2 * hi - 1),
        "logit_separation": logit_sep,
        "cohens_d": cohens_d,
    }

    print(f"\n{'='*65}")
    print(f"Param set : {param_set}   Model : {model_name}   N_test : {len(labels):,}")
    if shuffle_labels:
        print(f"*** PERMUTATION TEST: Labels shuffled ***")
    print(f"Logit Sep : {logit_sep:.4f}   Cohen's d : {cohens_d:.4f}")
    print(f"{'='*65}")
    _print_metrics(f"{model_name} (ours)", mean, lo, hi)

    # χ² baseline (no train data needed)
    A_test = test_ds._a.astype(np.float32)
    B_test = test_ds._b.astype(np.float32)
    y_test = test_ds._labels.astype(int)

    B_lwe  = B_test[y_test == 1]
    B_unif = B_test[y_test == 0]
    if len(B_lwe) > 0 and len(B_unif) > 0:
        y_chi2 = np.concatenate([np.zeros(len(B_unif)), np.ones(len(B_lwe))])
        scores_chi2 = np.concatenate([
            np.abs(B_unif.astype(float) % params.q - params.q / 2).mean(axis=-1),
            np.abs(B_lwe.astype(float)  % params.q - params.q / 2).mean(axis=-1),
        ])
        cm, clo, chi = bootstrap_auroc(y_chi2, scores_chi2, n_boot=n_boot)
        _print_metrics("χ² statistical", cm, clo, chi)
        results.update({"chi2_auroc": float(cm), "chi2_adv": 2*float(cm)-1})

    # Supervised baselines (need train data)
    if train_dir is not None:
        _, train_ds = _build_loader(model_name, train_dir, params, batch_size, repr_type=repr_type)
        A_train = train_ds._a.astype(np.float32)
        B_train = train_ds._b.astype(np.float32)
        y_train = train_ds._labels.astype(int)
        
        if shuffle_labels:
            np.random.shuffle(y_train)

        try:
            lr_res = run_logistic_regression(A_train, B_train, y_train,
                                             A_test,  B_test,  y_test)
            lm, llo, lhi = bootstrap_auroc(
                y_test, lr_res["model"].predict_proba(
                    np.concatenate([A_test.reshape(len(A_test), -1),
                                    B_test.reshape(len(B_test), -1)], axis=1)
                )[:, 1], n_boot=n_boot,
            )
            _print_metrics("Logistic regression", lm, llo, lhi)
            results.update({"lr_auroc": float(lm), "lr_adv": 2*float(lm)-1})
        except Exception as exc:
            print(f"  [LR baseline failed: {exc}]")

        try:
            mlp_res = run_mlp(A_train, B_train, y_train, A_test, B_test, y_test)
            mm, mlo, mhi = bootstrap_auroc(
                y_test, mlp_res["model"].predict_proba(
                    np.concatenate([A_test.reshape(len(A_test), -1),
                                    B_test.reshape(len(B_test), -1)], axis=1)
                )[:, 1], n_boot=n_boot,
            )
            _print_metrics("MLP", mm, mlo, mhi)
            results.update({"mlp_auroc": float(mm), "mlp_adv": 2*float(mm)-1})
        except Exception as exc:
            print(f"  [MLP baseline failed: {exc}]")

    print(f"{'='*55}\n")
    return results


if __name__ == "__main__":
    args = parse_args()

    # Delayed imports to allow `--help` to run instantly
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from latticeprobe.baselines import (
        bootstrap_auroc,
        chi2_distinguisher,
        run_logistic_regression,
        run_mlp,
    )
    from latticeprobe.datasets import LWEGraphDataset, LWESequenceDataset
    from latticeprobe.models.gnn import LWEGNN
    from latticeprobe.models.transformer import LWETransformer

    results = evaluate_model(
        checkpoint=args.checkpoint,
        model_name=args.model,
        param_set=args.param_set,
        test_dir=args.test_dir,
        train_dir=args.train_dir,
        batch_size=args.batch_size,
        n_boot=args.n_boot,
        device_arg=args.device,
        shuffle_labels=args.shuffle_labels,
        repr_type=args.repr,
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"Results written to {out}")
