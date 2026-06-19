#!/usr/bin/env python3
"""
Training pipeline for LatticeProbe Transformer and GNN models (paper §5).

Usage:
    python scripts/train.py \\
        --param-set ML-KEM-512 \\
        --model transformer \\
        --train-dir data/ML-KEM-512/train \\
        --val-dir   data/ML-KEM-512/val \\
        --output-dir checkpoints/transformer-512
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latticeprobe.params import PARAMS, get_params


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a LatticeProbe model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--param-set",    required=True, choices=list(PARAMS))
    p.add_argument("--model",        required=True, choices=["transformer", "gnn"])
    p.add_argument("--train-dir",    required=True)
    p.add_argument("--val-dir",      required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience",     type=int,   default=5,
                   help="Early-stopping patience (epochs without val AUROC improvement)")
    p.add_argument("--ckpt-every",   type=int,   default=5,
                   help="Save a periodic checkpoint every N epochs")
    p.add_argument("--device",       default="auto", help="cuda / cpu / auto")
    # Transformer size overrides (paper defaults used when omitted)
    p.add_argument("--d-model",    type=int, default=512)
    p.add_argument("--nhead",      type=int, default=8)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--ff-dim",     type=int, default=2048,
                   help="Transformer FFN inner dim (default 4×d_model=2048)")
    # GNN size override
    p.add_argument("--hidden",     type=int, default=256)
    p.add_argument("--gnn-layers", type=int, default=6)
    p.add_argument("--wandb",        action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-project", default="latticeprobe")
    p.add_argument("--compute-log",  default="compute_log.csv",
                   help="CSV file for GPU-hour tracking")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="Randomly permute training labels for permutation tests")
    p.add_argument("--repr", default="coeff", choices=["coeff", "ntt", "dual"],
                   help="Representation domain of the polynomials")
    return p.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(model_name: str, params, device: torch.device, args=None) -> nn.Module:
    if model_name == "transformer":
        kwargs = {}
        if args is not None:
            kwargs = dict(
                d_model=getattr(args, "d_model", 512),
                nhead=getattr(args, "nhead", 8),
                num_layers=getattr(args, "num_layers", 8),
                dim_feedforward=getattr(args, "ff_dim", 2048),
            )
        return LWETransformer(params, **kwargs).to(device)
    kwargs = {}
    if args is not None:
        kwargs = dict(
            hidden=getattr(args, "hidden", 256),
            num_layers=getattr(args, "gnn_layers", 6),
        )
    return LWEGNN(params, **kwargs).to(device)


def build_loader(model_name: str, data_dir: str, params, batch_size: int, shuffle: bool, repr_type: str = "coeff"):
    if model_name == "transformer":
        ds = LWESequenceDataset(data_dir, params, repr_type=repr_type)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                          pin_memory=False)
    else:
        from torch_geometric.loader import DataLoader as GeoLoader
        ds = LWEGraphDataset(data_dir, params, repr_type=repr_type)
        return GeoLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _append_compute_log(csv_path: str, row: dict) -> None:
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ── Training / Validation loop ────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    model_name: str,
    optimizer: AdamW | None = None,
    shuffle_labels: bool = False,
) -> tuple[float, float]:
    """
    Run one train (optimizer≠None) or validation epoch.
    Returns (mean_loss, auroc).
    """
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            if model_name == "transformer":
                tokens, labels = batch
                tokens = tokens.to(device)
                labels = labels.float().to(device).unsqueeze(1)   # (B,1)
                logits = model(tokens)                             # (B,1)
            else:
                data, labels = batch
                data   = data.to(device)
                labels = labels.float().to(device).reshape(-1, 1) # (B,1)
                logits = model(data)                               # (B,1)

            if training and shuffle_labels:
                idx = torch.randperm(labels.size(0))
                labels = labels[idx]

            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = labels.shape[0]
            total_loss += loss.item() * n
            all_logits.append(logits.detach().cpu().float().squeeze(1).numpy())
            all_labels.append(labels.detach().cpu().float().squeeze(1).numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    mean_loss  = total_loss / max(len(labels_np), 1)
    try:
        auroc = float(roc_auc_score(labels_np, logits_np))
    except ValueError:
        auroc = float("nan")
    return mean_loss, auroc


# ── Main training loop ────────────────────────────────────────────────────────

def train(args) -> Path:
    """
    Full training run. Returns the path to the best checkpoint.
    Can be called with a namespace produced by parse_args() or a compatible object.
    """
    params   = get_params(args.param_set)
    device   = resolve_device(args.device)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional W&B
    wandb_run = None
    if getattr(args, "wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            print("wandb not installed — skipping W&B logging")

    model = build_model(args.model, params, device, args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model : {args.model}  |  params : {n_params:,}  |  device : {device}")

    train_loader = build_loader(args.model, args.train_dir, params,
                                args.batch_size, shuffle=True, repr_type=getattr(args, "repr", "coeff"))
    val_loader   = build_loader(args.model, args.val_dir,   params,
                                args.batch_size, shuffle=False, repr_type=getattr(args, "repr", "coeff"))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_auroc    = -1.0
    patience_left = args.patience
    best_ckpt     = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_auroc = run_epoch(model, train_loader, device, args.model, optimizer, getattr(args, "shuffle_labels", False))
        val_loss,   val_auroc   = run_epoch(model, val_loader,   device, args.model, None)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_loss:.4f}  train_auroc={train_auroc:.4f}"
            f"  val_loss={val_loss:.4f}  val_auroc={val_auroc:.4f}"
            f"  {elapsed:.1f}s"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss, "train_auroc": train_auroc,
                "val_loss":   val_loss,   "val_auroc":   val_auroc,
                "lr": scheduler.get_last_lr()[0],
            })

        # Periodic checkpoint
        if epoch % args.ckpt_every == 0:
            ckpt = out_dir / f"ckpt_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch, "val_auroc": val_auroc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt)

        # Best checkpoint
        if val_auroc > best_auroc:
            best_auroc    = val_auroc
            patience_left = args.patience
            torch.save({
                "epoch": epoch, "val_auroc": val_auroc,
                "model_state": model.state_dict(),
                "args": vars(args),
            }, best_ckpt)
        else:
            patience_left -= 1

        # Compute log (GPU-hours = 0 when on CPU)
        gpu_h = elapsed / 3600.0 if device.type == "cuda" else 0.0
        _append_compute_log(args.compute_log, {
            "param_set":    args.param_set,
            "model":        args.model,
            "epoch":        epoch,
            "wall_seconds": round(elapsed, 2),
            "gpu_hours":    round(gpu_h, 6),
            "val_auroc":    round(val_auroc, 6),
        })

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"Best val AUROC: {best_auroc:.4f}  →  {best_ckpt}")
    if wandb_run is not None:
        wandb_run.finish()
    return best_ckpt


if __name__ == "__main__":
    args = parse_args()
    
    # Delayed imports to allow `--help` to run instantly
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from latticeprobe.datasets import LWEGraphDataset, LWESequenceDataset
    from latticeprobe.models.gnn import LWEGNN
    from latticeprobe.models.transformer import LWETransformer

    # Inject into global namespace for functions to use
    globals().update(locals())

    train(args)
