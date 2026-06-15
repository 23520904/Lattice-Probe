"""Tests for ring.py — NTT, INTT, and polynomial multiplication."""

import numpy as np
import pytest

from latticeprobe.ring import (
    N, Q,
    intt, ntt, ntt_mul, poly_add, poly_mul, poly_mul_naive,
)


RNG = np.random.default_rng(42)


def rand_poly() -> np.ndarray:
    return RNG.integers(0, Q, size=N, dtype=np.int64)


# ── NTT round-trip ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("_", range(10))
def test_ntt_roundtrip(_):
    f = rand_poly()
    assert np.array_equal(intt(ntt(f)), f % Q), "INTT(NTT(f)) must equal f"


def test_ntt_zero():
    z = np.zeros(N, dtype=np.int64)
    assert np.array_equal(ntt(z), z)
    assert np.array_equal(intt(z), z)


def test_ntt_one():
    """NTT of the polynomial 1 (constant) should give [1, 0, 1, 0, ...]."""
    e0 = np.zeros(N, dtype=np.int64)
    e0[0] = 1
    fhat = ntt(e0)
    # INTT should recover e0
    assert np.array_equal(intt(fhat), e0)


# ── Polynomial multiplication ─────────────────────────────────────────────────

@pytest.mark.parametrize("_", range(5))
def test_poly_mul_matches_naive(_):
    f, g = rand_poly(), rand_poly()
    ntt_result   = poly_mul(f, g)
    naive_result = poly_mul_naive(f, g)
    assert np.array_equal(ntt_result, naive_result), (
        f"NTT poly_mul disagrees with naive for seed iteration {_}"
    )


def test_poly_mul_commutative():
    f, g = rand_poly(), rand_poly()
    assert np.array_equal(poly_mul(f, g), poly_mul(g, f))


def test_poly_mul_zero():
    f = rand_poly()
    z = np.zeros(N, dtype=np.int64)
    assert np.array_equal(poly_mul(f, z), z)


def test_poly_mul_one():
    """Multiplying by the polynomial '1' is the identity."""
    f = rand_poly()
    one = np.zeros(N, dtype=np.int64)
    one[0] = 1
    assert np.array_equal(poly_mul(f, one), f % Q)


def test_poly_mul_distributive():
    """(f + g) * h == f*h + g*h  in R_q."""
    f, g, h = rand_poly(), rand_poly(), rand_poly()
    lhs = poly_mul(poly_add(f, g), h)
    rhs = poly_add(poly_mul(f, h), poly_mul(g, h))
    assert np.array_equal(lhs, rhs)


# ── Polynomial addition ───────────────────────────────────────────────────────

def test_poly_add_commutative():
    f, g = rand_poly(), rand_poly()
    assert np.array_equal(poly_add(f, g), poly_add(g, f))


def test_poly_add_zero():
    f = rand_poly()
    z = np.zeros(N, dtype=np.int64)
    assert np.array_equal(poly_add(f, z), f % Q)


def test_poly_add_in_range():
    f, g = rand_poly(), rand_poly()
    result = poly_add(f, g)
    assert result.min() >= 0 and result.max() < Q
