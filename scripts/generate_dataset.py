#!/usr/bin/env python3
"""
Generate sharded Module-LWE datasets for LatticeProbe training (paper §5).

Usage:
    python scripts/generate_dataset.py \\
        --param-set ML-KEM-512 \\
        --n-samples 65536 \\
        --output-dir data/ML-KEM-512/train

Each output shard is a .npz file containing float32 arrays (a, b) and int8 labels.
A fixed LWE secret is drawn once and saved to secret.npy in the output directory.
50% of samples are genuine LWE (label=1), 50% are uniform random (label=0).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running as a script from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latticeprobe.params import PARAMS, get_params


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate sharded LWE datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--param-set", required=True, choices=list(PARAMS),
                   help="Parameter set (e.g. ML-KEM-512, W3, edge)")
    p.add_argument("--n-samples", type=int, required=True,
                   help="Total samples to generate (e.g. 65536 = 2^16)")
    p.add_argument("--shard-size", type=int, default=4096,
                   help="Samples per .npz shard file")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write shard_*.npz and secret.npy")
    p.add_argument("--num-secrets", type=int, default=1,
                   help="Number of distinct secrets to generate and embed in this dataset")
    p.add_argument("--secret-file", type=str, default=None,
                   help="Path to an existing secrets.npy file to reuse (enforces same-key across splits)")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="Scale multiplier for the noise variance (e.g. 0.90 for 90 percent noise)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    return p.parse_args(argv)


def generate_dataset(
    param_set: str,
    n_samples: int,
    output_dir: str,
    shard_size: int = 4096,
    quiet: bool = False,
    noise_scale: float = 1.0,
    num_secrets: int = 1,
    secret_file: str | None = None,
) -> Path:
    """
    Generate and save a sharded dataset.

    A single LWE secret is sampled once and reused for all LWE samples so that
    all samples in a split share the same key — caller must use a different call
    (different output dir) for train vs. test to enforce disjoint key sets.

    Returns the output directory Path.
    """
    params = get_params(param_set)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if secret_file is not None:
        secrets = np.load(secret_file)
        # Ensure it has the 3D shape (N, k, n) expected by sampler
        if secrets.ndim == 2:
            secrets = secrets[np.newaxis, ...]
        np.save(out / "secrets.npy", secrets)
    else:
        # Draw the specified number of secrets for this split.
        from latticeprobe.prng import fresh_rng
        from latticeprobe.sampler import _sample_secret
        secrets = np.stack([_sample_secret(params, fresh_rng()) for _ in range(num_secrets)])
        np.save(out / "secrets.npy", secrets)

    from latticeprobe.datasets import save_shard
    from latticeprobe.sampler import generate_batch

    generated = 0
    shard_idx = 0
    bar = tqdm(total=n_samples, unit="samples", desc=param_set, disable=quiet)

    while generated < n_samples:
        batch = min(shard_size, n_samples - generated)
        A, B, labels = generate_batch(params, batch, secret=secrets, noise_scale=noise_scale)
        save_shard(str(out / f"shard_{shard_idx:05d}.npz"), A, B, labels)
        generated += batch
        shard_idx += 1
        bar.update(batch)

    bar.close()
    if not quiet:
        print(f"Wrote {shard_idx} shard(s) ({generated:,} samples) → {out}")
    return out


def main():
    args = parse_args()
    generate_dataset(
        param_set=args.param_set,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        quiet=args.quiet,
        noise_scale=args.noise_scale,
        num_secrets=args.num_secrets,
        secret_file=args.secret_file,
    )


if __name__ == "__main__":
    main()
