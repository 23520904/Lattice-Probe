"""
Classical baselines required by the paper (§4.3 Baselines).

Baseline 1: Logistic regression on raw flattened coefficient features.
Baseline 2: MLP on flattened representations.
Baseline 3: χ² statistical distinguisher on coefficient distributions.
Baseline 4: lattice-estimator security estimate (log2 attack cost).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from latticeprobe.params import LWEParams


def flatten(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Flatten (A, B) batch into a 2D feature matrix shape (N, k*n + n)."""
    N = A.shape[0]
    return np.concatenate([A.reshape(N, -1), B.reshape(N, -1)], axis=1).astype(np.float32)


# ── Baseline 1 — Logistic Regression ─────────────────────────────────────────

def run_logistic_regression(
    A_train: np.ndarray,
    B_train: np.ndarray,
    y_train: np.ndarray,
    A_test: np.ndarray,
    B_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit logistic regression, return AUROC on test set."""
    X_train = flatten(A_train, B_train)
    X_test  = flatten(A_test,  B_test)
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    return {"auroc": float(roc_auc_score(y_test, scores)), "model": clf}


# ── Baseline 2 — MLP ─────────────────────────────────────────────────────────

def run_mlp(
    A_train: np.ndarray,
    B_train: np.ndarray,
    y_train: np.ndarray,
    A_test: np.ndarray,
    B_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layer_sizes: tuple = (512, 256),
) -> dict:
    """Fit MLP classifier, return AUROC on test set."""
    X_train = flatten(A_train, B_train)
    X_test  = flatten(A_test,  B_test)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    return {"auroc": float(roc_auc_score(y_test, scores)), "model": clf}


# ── Baseline 3 — χ² Statistical Distinguisher ────────────────────────────────

def chi2_distinguisher(
    B_lwe: np.ndarray,
    B_unif: np.ndarray,
    params: LWEParams,
) -> dict:
    """
    χ² test on the marginal distribution of b coefficients.

    Theory: b coefficients of genuine LWE samples are *not* perfectly uniform
    (they have a slight bias from the noise distribution).  Under H0 (uniform),
    the coefficient histogram should match Uniform[0, q).

    Args:
        B_lwe:  float32/int64, shape (N_lwe, n) — b polynomials from LWE samples.
        B_unif: float32/int64, shape (N_unif, n) — b polynomials from uniform samples.
        params: LWEParams (provides q for bin count).

    Returns:
        dict with 'p_value_lwe', 'p_value_unif', 'auroc_proxy'.
    """
    q = params.q

    def _chi2_pvalue(B: np.ndarray) -> float:
        flat = B.reshape(-1).astype(int) % q
        counts, _ = np.histogram(flat, bins=q, range=(0, q))
        expected = np.full(q, flat.size / q)
        chi2_stat, p = stats.chisquare(counts, f_exp=expected)
        return float(p)

    p_lwe  = _chi2_pvalue(B_lwe)
    p_unif = _chi2_pvalue(B_unif)

    # As a proxy AUROC: compute per-coefficient p-values then use −log(p) as score.
    def _score(B: np.ndarray) -> np.ndarray:
        """Per-sample distinguishing score via average absolute deviation from q/2."""
        B_int = B.astype(float) % q
        return np.abs(B_int - q / 2).mean(axis=-1)  # higher = more structured

    scores_lwe  = _score(B_lwe)
    scores_unif = _score(B_unif)
    all_scores = np.concatenate([scores_unif, scores_lwe])
    all_labels = np.concatenate([np.zeros(len(B_unif)), np.ones(len(B_lwe))])
    auroc = float(roc_auc_score(all_labels, all_scores))

    return {"p_value_lwe": p_lwe, "p_value_unif": p_unif, "auroc": auroc}


# ── Baseline 4 — Lattice Estimator ───────────────────────────────────────────

def run_lattice_estimator(params: LWEParams) -> dict | None:
    """
    Call the lattice-estimator Python API to get the log2 cost of the best
    classical attack on the given LWE parameter set.

    Returns None if the lattice-estimator package is not installed.
    """
    try:
        from estimator import LWE, ND  # type: ignore[import]
    except ImportError:
        return None

    if params.noise == "cbd":
        Xe = ND.CenteredBinomial(params.eta)
        Xs = ND.CenteredBinomial(params.eta) if params.secret == "cbd" else ND.Uniform(0, 1)
    elif params.noise == "gaussian":
        Xe = ND.DiscreteGaussian(params.sigma)
        Xs = ND.DiscreteGaussian(params.sigma)
    else:
        return {"log2_cost": float("inf"), "note": "zero noise — trivially solvable"}

    cost = LWE.estimate(
        n=params.n * params.k,  # total dimension = k * n for Module-LWE
        q=params.q,
        Xs=Xs,
        Xe=Xe,
    )
    log2_costs = {k: float(v.rop.log(2)) for k, v in cost.items() if hasattr(v, "rop")}
    min_cost = min(log2_costs.values()) if log2_costs else float("nan")
    return {"log2_cost": min_cost, "per_attack": log2_costs}


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    95% BCa bootstrap confidence interval for AUROC.

    Returns:
        (mean_auroc, lower_bound, upper_bound)
    """
    def _auc(yt, ys, axis=-1):
        if yt.ndim == 1:
            if len(np.unique(yt)) < 2: return 0.5
            return roc_auc_score(yt, ys)
        else:
            # 2D case from scipy.stats.bootstrap
            res = []
            for i in range(yt.shape[0] if axis == 1 else yt.shape[1]):
                y_t = yt[i] if axis == 1 else yt[:, i]
                y_s = ys[i] if axis == 1 else ys[:, i]
                if len(np.unique(y_t)) < 2: res.append(0.5)
                else: res.append(roc_auc_score(y_t, y_s))
            return np.array(res)

    mean_auc = _auc(y_true, y_score)
    try:
        res = stats.bootstrap(
            (y_true, y_score),
            statistic=_auc,
            n_resamples=n_boot,
            confidence_level=ci,
            method='bca',
            paired=True,
            random_state=seed,
        )
        return float(mean_auc), float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception as e:
        print(f"  [BCa failed: {e}. Falling back to percentile.]")
        rng = np.random.default_rng(seed)
        aucs = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yt, ys = y_true[idx], y_score[idx]
            if len(np.unique(yt)) < 2:
                continue
            aucs.append(roc_auc_score(yt, ys))
        aucs = np.array(aucs)
        alpha = (1 - ci) / 2
        return float(mean_auc), float(np.percentile(aucs, alpha * 100)), float(np.percentile(aucs, (1 - alpha) * 100))
