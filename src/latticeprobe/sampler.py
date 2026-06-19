"""
Module-LWE sample generator.

Control invariants (paper §5.1):
- fresh_rng() is called at the start of every sample — no shared state.
- No caching or deterministic shortcuts.
- For CBD noise: exact FIPS-203 formula.
- For Gaussian noise: discrete Gaussian via rounded Normal + rejection (tails clipped at 6σ).
"""

import numpy as np

from latticeprobe.params import LWEParams
from latticeprobe.prng import fresh_rng
from latticeprobe.ring import poly_add, poly_mul


# ── Noise samplers ────────────────────────────────────────────────────────────

def cbd(eta: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Centred binomial distribution over n coefficients.

    Each coefficient = sum(a_i) - sum(b_i) where a_i, b_i ~ Bernoulli(1/2),
    i = 1..eta. Result in [-eta, eta], returned reduced to [0, q) externally.
    """
    bits = rng.integers(0, 2, size=(n, 2 * eta), dtype=np.int64)
    return bits[:, :eta].sum(axis=1) - bits[:, eta:].sum(axis=1)


def discrete_gaussian(sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Discrete Gaussian: round a continuous N(0, σ²) sample to the nearest integer.
    Tails beyond 6σ are resampled (probability < 10^-9 per coefficient).
    """
    bound = max(1, int(np.ceil(6 * sigma)))
    result = np.zeros(n, dtype=np.int64)
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        m = remaining.sum()
        samples = np.round(rng.normal(0, sigma, size=m)).astype(np.int64)
        clipped = np.abs(samples) <= bound
        idxs = np.where(remaining)[0][clipped]
        result[idxs] = samples[clipped]
        remaining[idxs] = False
    return result


def _sample_noise(params: LWEParams, rng: np.random.Generator, noise_scale: float = 1.0) -> np.ndarray:
    """Sample one noise polynomial according to params."""
    if params.noise == "cbd":
        if noise_scale != 1.0:
            var = (params.eta / 2) * (noise_scale ** 2)
            return discrete_gaussian(np.sqrt(var), params.n, rng)
        return cbd(params.eta, params.n, rng)
    if params.noise == "gaussian":
        return discrete_gaussian(params.sigma * noise_scale, params.n, rng)
    if params.noise == "zero":
        return np.zeros(params.n, dtype=np.int64)
    raise ValueError(f"Unknown noise type: {params.noise}")


def _sample_secret(params: LWEParams, rng: np.random.Generator) -> np.ndarray:
    """Sample one secret polynomial according to params."""
    if params.secret == "binary":
        return rng.integers(0, 2, size=(params.k, params.n), dtype=np.int64)
    if params.secret == "ternary":
        return rng.integers(-1, 2, size=(params.k, params.n), dtype=np.int64)
    if params.secret == "uniform":
        return rng.integers(0, params.q, size=(params.k, params.n), dtype=np.int64)
    # Default: CBD with same eta as noise
    return np.stack([cbd(params.eta if params.noise == "cbd" else 1,
                         params.n, rng)
                     for _ in range(params.k)])


# ── Sample generators ─────────────────────────────────────────────────────────

def generate_lwe_sample(
    params: LWEParams,
    secret: np.ndarray | None = None,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one Module-LWE sample (a, b, s).

    Returns:
        a: shape (k, n), each row a polynomial in R_q (coefficients in [0, q))
        b: shape (n,), polynomial b = <a,s> + e  mod q
        s: shape (k, n), secret (same for all samples from one split — caller's
           responsibility to keep a fixed secret across samples in a split)
    """
    rng = fresh_rng()
    q, n, k = params.q, params.n, params.k

    a = rng.integers(0, q, size=(k, n), dtype=np.int64)

    if secret is None:
        s = _sample_secret(params, rng)
    else:
        s = np.asarray(secret, dtype=np.int64)

    e = _sample_noise(params, rng, noise_scale=noise_scale)

    # b = e + sum_r  poly_mul(a[r], s[r])  mod q
    b = e.copy()
    for r in range(k):
        b = poly_add(b, poly_mul(a[r], s[r] % q))

    return a, b % q, s


def generate_uniform_sample(params: LWEParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one uniform random sample (a, b) — b has no algebraic relation to a.

    Returns:
        a: shape (k, n), uniform in [0, q)
        b: shape (n,),   uniform in [0, q)  (independent of a)
    """
    rng = fresh_rng()
    q, n, k = params.q, params.n, params.k
    a = rng.integers(0, q, size=(k, n), dtype=np.int64)
    b = rng.integers(0, q, size=(n,), dtype=np.int64)
    return a, b


# ── Batch generation ──────────────────────────────────────────────────────────

def generate_batch(
    params: LWEParams,
    n_samples: int,
    secret: np.ndarray | None = None,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a mixed batch of LWE and uniform samples.

    Each sample is independently labelled 1 (LWE) or 0 (uniform) with equal
    probability. The `secret` parameter can be a single secret (shape `(k, n)`) or
    a bank of secrets (shape `(N, k, n)`). For each LWE sample, one secret is
    drawn uniformly from the bank.

    Returns:
        A: float32, shape (n_samples, k, n)
        B: float32, shape (n_samples, n)
        labels: int8, shape (n_samples,)  — 1=LWE, 0=uniform
    """
    rng_meta = fresh_rng()
    labels = rng_meta.integers(0, 2, size=n_samples, dtype=np.int8)

    # Normalize secret to shape (N, k, n)
    if secret is None:
        _, _, sec_single = generate_lwe_sample(params, noise_scale=noise_scale)
        secret = sec_single[np.newaxis, ...]
    elif secret.ndim == 2:
        secret = secret[np.newaxis, ...]

    A = np.empty((n_samples, params.k, params.n), dtype=np.float32)
    B = np.empty((n_samples, params.n), dtype=np.float32)

    for i in range(n_samples):
        if labels[i] == 1:
            sec = secret[rng_meta.integers(0, len(secret))]
            a, b, _ = generate_lwe_sample(params, secret=sec, noise_scale=noise_scale)
        else:
            a, b = generate_uniform_sample(params)
        A[i] = a
        B[i] = b

    return A, B, labels
