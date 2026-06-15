"""Parameter sets for LatticeProbe (FIPS-203 §1.3 + weakened/edge-of-margin regimes)."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LWEParams:
    name: str
    n: int
    k: int
    q: int
    # Noise distribution: "cbd" uses eta; "gaussian" uses sigma; "zero" means no noise.
    noise: Literal["cbd", "gaussian", "zero"]
    eta: int = 0               # CBD parameter (used when noise="cbd")
    sigma: float = 0.0         # Gaussian σ  (used when noise="gaussian")
    secret: Literal["cbd", "binary"] = "cbd"  # secret distribution


PARAMS: dict[str, LWEParams] = {
    # ── Standardised (FIPS-203 exact) ────────────────────────────────────────
    "ML-KEM-512": LWEParams(
        name="ML-KEM-512", n=256, k=2, q=3329, noise="cbd", eta=3,
    ),
    "ML-KEM-768": LWEParams(
        name="ML-KEM-768", n=256, k=3, q=3329, noise="cbd", eta=2,
    ),
    "ML-KEM-1024": LWEParams(
        name="ML-KEM-1024", n=256, k=4, q=3329, noise="cbd", eta=2,
    ),
    # ── Weakened — sanity checks ──────────────────────────────────────────────
    # W1: binary secret, same noise as ML-KEM-512 (known Arora-Ge vulnerable)
    "W1": LWEParams(
        name="W1", n=256, k=2, q=3329, noise="cbd", eta=3, secret="binary",
    ),
    # W2: no noise — trivially solvable by Gaussian elimination
    "W2": LWEParams(
        name="W2", n=256, k=2, q=3329, noise="zero",
    ),
    # W3: σ reduced 60% below ML-KEM-512 baseline (σ_512 ≈ 1.225 → 0.490)
    "W3": LWEParams(
        name="W3", n=256, k=2, q=3329, noise="gaussian", sigma=0.490,
    ),
    # ── Edge-of-margin ────────────────────────────────────────────────────────
    # σ reduced 35% below ML-KEM-512 (1.225 × 0.65 ≈ 0.796)
    "edge": LWEParams(
        name="edge", n=256, k=2, q=3329, noise="gaussian", sigma=0.796,
    ),
}


def get_params(name: str) -> LWEParams:
    if name not in PARAMS:
        raise ValueError(f"Unknown parameter set '{name}'. Choose from: {list(PARAMS)}")
    return PARAMS[name]
