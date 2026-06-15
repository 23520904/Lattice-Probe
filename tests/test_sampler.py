"""Tests for sampler.py — LWE structure, noise distributions, and controls."""

import numpy as np
import pytest
from scipy import stats

from latticeprobe.params import get_params
from latticeprobe.ring import poly_add, poly_mul, Q, N
from latticeprobe.sampler import (
    cbd, discrete_gaussian, generate_batch,
    generate_lwe_sample, generate_uniform_sample,
)


# ── CBD distribution tests ────────────────────────────────────────────────────

@pytest.mark.parametrize("eta", [1, 2, 3])
def test_cbd_range(eta):
    from latticeprobe.prng import fresh_rng
    rng = fresh_rng()
    samples = cbd(eta, 256, rng)
    assert samples.min() >= -eta, "CBD must be ≥ -eta"
    assert samples.max() <= eta,  "CBD must be ≤ +eta"


@pytest.mark.parametrize("eta", [2, 3])
def test_cbd_distribution(eta):
    """χ² test: CBD(η) PMF should match the theoretical distribution."""
    from latticeprobe.prng import fresh_rng
    n_samples = 50_000
    vals = []
    for _ in range(n_samples // 256):
        rng = fresh_rng()
        vals.extend(cbd(eta, 256, rng).tolist())
    vals = np.array(vals[:n_samples])

    # Theoretical PMF of CBD(eta)
    support = np.arange(-eta, eta + 1)
    from math import comb
    pmf = np.array([
        sum(comb(eta, i) * comb(eta, i - v) for i in range(max(0, v), eta + 1))
        / (4 ** eta)
        for v in support
    ])

    counts = np.array([(vals == v).sum() for v in support], dtype=float)
    expected = pmf * counts.sum()   # normalise to actual sample count
    _, p = stats.chisquare(counts, f_exp=expected)
    assert p > 0.01, f"CBD({eta}) failed χ² test: p={p:.4f}"


# ── Discrete Gaussian tests ───────────────────────────────────────────────────

def test_discrete_gaussian_shape():
    from latticeprobe.prng import fresh_rng
    rng = fresh_rng()
    s = discrete_gaussian(0.796, 256, rng)
    assert s.shape == (256,)


def test_discrete_gaussian_mean():
    from latticeprobe.prng import fresh_rng
    samples = np.concatenate([discrete_gaussian(1.0, 256, fresh_rng()) for _ in range(200)])
    assert abs(samples.mean()) < 0.1, "Discrete Gaussian should have mean ≈ 0"


# ── LWE structure verification ────────────────────────────────────────────────

@pytest.mark.parametrize("param_name", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
def test_lwe_structure(param_name):
    """
    Verify b = <a, s> + e  (mod q) holds for a generated sample.
    This directly tests that the LWE generator implements the correct structure.
    """
    params = get_params(param_name)
    a, b, s = generate_lwe_sample(params)

    # Reconstruct <a, s> mod q
    inner = np.zeros(N, dtype=np.int64)
    for r in range(params.k):
        inner = poly_add(inner, poly_mul(a[r], s[r] % Q))

    # e = b - <a, s>  mod q
    e = (b.astype(np.int64) - inner + Q) % Q
    # Centred: map to [-q/2, q/2]
    e_centred = np.where(e > Q // 2, e - Q, e)

    assert np.all(np.abs(e_centred) <= params.eta), (
        f"Error coefficients out of range for {param_name}: "
        f"max abs = {np.abs(e_centred).max()}, expected ≤ {params.eta}"
    )


def test_lwe_fixed_secret():
    """With a fixed secret, multiple samples should all satisfy the LWE relation."""
    params = get_params("ML-KEM-512")
    _, _, s = generate_lwe_sample(params)

    for _ in range(5):
        a, b, _ = generate_lwe_sample(params, secret=s)
        inner = np.zeros(N, dtype=np.int64)
        for r in range(params.k):
            inner = poly_add(inner, poly_mul(a[r], s[r] % Q))
        e = (b.astype(np.int64) - inner + Q) % Q
        e_centred = np.where(e > Q // 2, e - Q, e)
        assert np.all(np.abs(e_centred) <= params.eta)


# ── Uniform sample tests ──────────────────────────────────────────────────────

def test_uniform_sample_range():
    params = get_params("ML-KEM-512")
    for _ in range(20):
        a, b = generate_uniform_sample(params)
        assert a.min() >= 0 and a.max() < Q
        assert b.min() >= 0 and b.max() < Q


def test_uniform_b_is_uniform():
    """KS test: b from uniform samples should be Uniform[0, Q)."""
    params = get_params("ML-KEM-512")
    b_vals = np.concatenate([generate_uniform_sample(params)[1] for _ in range(50)])
    _, p = stats.kstest(b_vals, "uniform", args=(0, Q))
    assert p > 0.01, f"Uniform sample b failed KS test: p={p:.4f}"


# ── W2 sanity check ───────────────────────────────────────────────────────────

def test_w2_no_noise():
    """W2: noise=zero means b = <a, s> exactly."""
    params = get_params("W2")
    a, b, s = generate_lwe_sample(params)
    inner = np.zeros(N, dtype=np.int64)
    for r in range(params.k):
        inner = poly_add(inner, poly_mul(a[r], s[r] % Q))
    assert np.array_equal(b, inner % Q), "W2: b must equal <a,s> with no noise"


# ── Batch generation ──────────────────────────────────────────────────────────

def test_batch_shape():
    params = get_params("ML-KEM-512")
    A, B, labels = generate_batch(params, n_samples=16)
    assert A.shape == (16, params.k, params.n)
    assert B.shape == (16, params.n)
    assert labels.shape == (16,)


def test_batch_labels_binary():
    params = get_params("ML-KEM-512")
    _, _, labels = generate_batch(params, n_samples=64)
    assert set(labels.tolist()).issubset({0, 1})
