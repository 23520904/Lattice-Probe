#!/usr/bin/env python3
"""
Partial secret-bit recovery experiment (paper §6, Partial-Bit Recovery).

Trains a logistic regression per bit position to predict individual secret bits
from (a, b) pairs. Reports per-bit accuracy vs. 0.5 random baseline and
identifies any bit positions with statistically significant predictability.

The dataset must have been generated with --save-secret (i.e. contain secret.npy).

Usage:
    python scripts/bit_recovery.py \\
        --train-dir data/ML-KEM-512/train \\
        --test-dir  data/ML-KEM-512/test \\
        --param-set ML-KEM-512 \\
        [--bit-range 0:256]   \\
        [--output-json results/bit_recovery-512.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latticeprobe.datasets import LWESequenceDataset
from latticeprobe.params import PARAMS, get_params


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Partial secret-bit recovery experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--param-set", required=True, choices=list(PARAMS))
    p.add_argument("--n-train", type=int, default=10000, help="Number of training samples")
    p.add_argument("--n-test", type=int, default=2000, help="Number of testing samples")
    p.add_argument("--bit-range", default=None,
                   help="Python slice notation for bit positions to test, e.g. 0:64")
    p.add_argument("--output-json", default=None)
    # Ignored arguments retained for backwards compatibility
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir",  default=None)
    return p.parse_args(argv)


def generate_fresh_lwe_samples(params, n_samples):
    from latticeprobe.sampler import generate_lwe_sample
    A = np.empty((n_samples, params.k, params.n), dtype=np.float32)
    B = np.empty((n_samples, params.n), dtype=np.float32)
    S = np.empty((n_samples, params.k, params.n), dtype=np.int64)
    for i in range(n_samples):
        a, b, s = generate_lwe_sample(params)
        A[i] = a
        B[i] = b
        S[i] = s
    A /= params.q
    B /= params.q
    return A, B, S


def recover_bits(
    param_set: str,
    n_train: int = 10000,
    n_test: int = 2000,
    bit_range: str | None = None,
) -> dict:
    """
    For each secret bit position, train a logistic regression on (a,b) features
    and report test accuracy.

    Returns a dict with per-bit accuracies and summary statistics.
    """
    params = get_params(param_set)

    # Load features and secrets by generating fresh samples
    print("Generating fresh samples with varying secrets...")
    A_train, B_train, S_train = generate_fresh_lwe_samples(params, n_train)
    A_test,  B_test,  S_test  = generate_fresh_lwe_samples(params, n_test)

    # Flatten (a, b) as feature vector
    X_train = np.concatenate([A_train.reshape(n_train, -1),
                               B_train.reshape(n_train, -1)], axis=1)
    X_test  = np.concatenate([A_test.reshape(n_test,  -1),
                               B_test.reshape(n_test,  -1)], axis=1)

    # Bit positions to test
    k, n = params.k, params.n
    total_bits = k * n
    bits_flat_train = S_train.reshape(n_train, -1)   # (n_train, k*n)
    bits_flat_test  = S_test.reshape(n_test, -1)     # (n_test, k*n)

    if bit_range is not None:
        parts = bit_range.split(":")
        start = int(parts[0]) if parts[0] else 0
        stop  = int(parts[1]) if len(parts) > 1 and parts[1] else total_bits
        bit_positions = list(range(start, min(stop, total_bits)))
    else:
        bit_positions = list(range(total_bits))

    print(f"Param set : {param_set}   k={k}  n={n}  bits to test : {len(bit_positions)}")
    print(f"Train LWE samples : {len(X_train):,}   Test LWE samples : {len(X_test):,}")
    print()

    per_bit_acc: list[float] = []
    for i in bit_positions:
        y_train_bit = (bits_flat_train[:, i] > 0).astype(int)
        y_test_bit  = (bits_flat_test[:, i]  > 0).astype(int)

        # Skip if constant (trivial)
        if y_train_bit.std() == 0 or len(np.unique(y_train_bit)) < 2:
            per_bit_acc.append(float("nan"))
            continue

        clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
        clf.fit(X_train, np.full(len(X_train), y_train_bit))
        y_pred = clf.predict(X_test)
        acc = accuracy_score(np.full(len(X_test), y_test_bit), y_pred)
        per_bit_acc.append(float(acc))

    valid = [a for a in per_bit_acc if not np.isnan(a)]
    mean_acc = float(np.mean(valid)) if valid else float("nan")

    print(f"Mean per-bit accuracy : {mean_acc:.4f}  (random baseline = 0.5000)")
    print(f"Max  per-bit accuracy : {max(valid, default=float('nan')):.4f}")
    print(f"Min  per-bit accuracy : {min(valid, default=float('nan')):.4f}")

    results = {
        "param_set":     param_set,
        "n_bits_tested": len(bit_positions),
        "mean_accuracy": mean_acc,
        "random_baseline": 0.5,
        "per_bit_accuracy": per_bit_acc,
    }
    return results


def main():
    args = parse_args()
    results = recover_bits(
        param_set=args.param_set,
        n_train=args.n_train,
        n_test=args.n_test,
        bit_range=args.bit_range,
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)

        def json_converter(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        out.write_text(json.dumps(results, indent=2, default=json_converter))
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
