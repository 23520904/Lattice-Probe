"""
Polynomial ring arithmetic for R_q = Z_q[X]/(X^n + 1).

Implements the negacyclic NTT following FIPS-203 §4.3 exactly:
  - q = 3329, n = 256, primitive root ζ = 17
  - q - 1 = 3328 = 2^8 × 13, so a 256th root of unity exists mod q
  - NTT maps to 128 residue pairs (incomplete / length-256 negacyclic NTT)
"""

import numpy as np

Q: int = 3329
N: int = 256
ZETA: int = 17


def _bit_rev7(k: int) -> int:
    """7-bit bit reversal of k."""
    r = 0
    for _ in range(7):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


# ZETAS[k] = 17^BitRev(7,k) mod 3329  for k = 0..127  (NTT twiddle factors)
ZETAS: np.ndarray = np.array([pow(ZETA, _bit_rev7(k), Q) for k in range(128)], dtype=np.int64)

# GAMMAS[i] = 17^(2*BitRev(7,i)+1) mod 3329  for i = 0..127  (base-case mul)
GAMMAS: np.ndarray = np.array([pow(ZETA, 2 * _bit_rev7(i) + 1, Q) for i in range(128)], dtype=np.int64)

# 128^{-1} mod 3329  (needed by INTT: 7 butterfly layers accumulate factor 2^7 = 128)
_INTT_FACTOR: int = pow(128, -1, Q)  # = 3303


def ntt(f: np.ndarray) -> np.ndarray:
    """
    Negacyclic NTT (FIPS-203 Algorithm 41).

    Args:
        f: int64 array of length 256, coefficients in [0, q).
    Returns:
        fhat: int64 array of length 256 in NTT domain.
    """
    fhat = np.array(f, dtype=np.int64) % Q
    k = 1
    length = 128
    while length >= 2:
        n_groups = N // (2 * length)
        # Reshape to (n_groups, 2, length): view[g, 0, :]=top, view[g, 1, :]=bot
        view = fhat.reshape(n_groups, 2, length)
        top = view[:, 0, :].copy()
        bot = view[:, 1, :].copy()
        zetas = ZETAS[k: k + n_groups, np.newaxis]   # (n_groups, 1)
        t = zetas * bot % Q
        view[:, 0, :] = (top + t) % Q
        view[:, 1, :] = (top - t + Q) % Q
        k += n_groups
        length //= 2
    return fhat


def intt(fhat: np.ndarray) -> np.ndarray:
    """
    Inverse negacyclic NTT (FIPS-203 Algorithm 42).

    Args:
        fhat: int64 array of length 256 in NTT domain.
    Returns:
        f: int64 array of length 256, coefficients in [0, q).
    """
    f = np.array(fhat, dtype=np.int64) % Q
    k = 127
    length = 2
    while length <= 128:
        n_groups = N // (2 * length)
        view = f.reshape(n_groups, 2, length)
        top = view[:, 0, :].copy()
        bot = view[:, 1, :].copy()
        # Use positive ZETAS[k] in decreasing order (empirically correct and consistent
        # with the pqcrystals C reference; FIPS-203 Alg 42 notation uses "-zeta" due to
        # a different implicit sign convention, but the C reference does not negate).
        zetas = ZETAS[k - n_groups + 1: k + 1][::-1, np.newaxis]  # (n_groups, 1)
        view[:, 0, :] = (top + bot) % Q
        view[:, 1, :] = zetas * (bot - top + Q) % Q
        k -= n_groups
        length *= 2
    return f * _INTT_FACTOR % Q


def ntt_mul(fhat: np.ndarray, ghat: np.ndarray) -> np.ndarray:
    """
    Pointwise multiplication in the NTT domain (FIPS-203 Algorithm 11).

    Each pair (fhat[2i], fhat[2i+1]) lives in Z_q[X]/(X^2 - gamma_i).
    """
    a = fhat.reshape(128, 2).astype(np.int64)
    b = ghat.reshape(128, 2).astype(np.int64)
    a0, a1 = a[:, 0], a[:, 1]
    b0, b1 = b[:, 0], b[:, 1]
    g = GAMMAS
    hhat = np.empty(N, dtype=np.int64)
    hhat[0::2] = (a0 * b0 + a1 * b1 * g) % Q
    hhat[1::2] = (a0 * b1 + a1 * b0) % Q
    return hhat


def poly_mul(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Polynomial multiplication in R_q via NTT."""
    return intt(ntt_mul(ntt(f), ntt(g)))


def poly_add(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Polynomial addition in R_q."""
    return (np.asarray(f, dtype=np.int64) + np.asarray(g, dtype=np.int64)) % Q


def poly_mul_naive(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Naive O(n²) poly multiplication in R_q — for testing only."""
    f, g = np.asarray(f, dtype=np.int64), np.asarray(g, dtype=np.int64)
    result = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for j in range(N):
            coeff = f[i] * g[j] % Q
            idx = i + j
            if idx < N:
                result[idx] = (result[idx] + coeff) % Q
            else:
                result[idx - N] = (result[idx - N] - coeff + Q) % Q
    return result
