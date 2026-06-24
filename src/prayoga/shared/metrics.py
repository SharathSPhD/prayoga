"""Statistics for prayoga gates — CPU-only (numpy/scipy).

Two layers:
  * Reusable correlation/bootstrap/power helpers ported from
    ActiveCircuitDiscovery `src/core/metrics.py` (kept ~parity).
  * Generic gate primitives every closure gate needs:
    ``bootstrap_ci``, ``permutation_test``, ``cohens_d``, ``label_shuffle_null``.

The ``label_shuffle_null`` primitive is the enforced null control for every
Axis-C probe (objectives §0 non-triviality bar).

Claim-tier: plumbing; no claim of its own.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

# --------------------------------------------------------------------------- #
# Generic gate primitives (prayoga)
# --------------------------------------------------------------------------- #


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d between two independent samples (pooled SD)."""
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    na, nb = len(av), len(bv)
    if na < 2 or nb < 2:
        return 0.0
    pooled_var = ((na - 1) * av.var(ddof=1) + (nb - 1) * bv.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return 0.0
    return float((av.mean() - bv.mean()) / np.sqrt(pooled_var))


def bootstrap_ci(
    x: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for an arbitrary statistic of one sample."""
    xv = np.asarray(x, dtype=float)
    if len(xv) == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(random_state)
    boot = np.empty(n_boot)
    n = len(xv)
    for i in range(n_boot):
        boot[i] = stat(xv[rng.randint(0, n, n)])
    return (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_perm: int = 10_000,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    alternative: str = "two-sided",
    random_state: int = 42,
) -> float:
    """Two-sample permutation test p-value (default statistic = mean difference)."""
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if statistic is None:
        statistic = lambda x, y: float(x.mean() - y.mean())  # noqa: E731
    observed = statistic(av, bv)
    pooled = np.concatenate([av, bv])
    na = len(av)
    rng = np.random.RandomState(random_state)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm = statistic(pooled[:na], pooled[na:])
        if alternative == "two-sided":
            count += abs(perm) >= abs(observed)
        elif alternative == "greater":
            count += perm >= observed
        else:  # "less"
            count += perm <= observed
    return float((count + 1) / (n_perm + 1))


def label_shuffle_null(
    fit_score: Callable[[np.ndarray, np.ndarray], float],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_shuffle: int = 1_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Label-shuffled null distribution for a probe score.

    ``fit_score(X, y)`` trains a probe on (X, y) and returns a scalar score
    (e.g. held-out accuracy). The labels are permuted ``n_shuffle`` times to
    build the null; the empirical p-value is the fraction of null scores >=
    the true score. This is the mandatory null control for Axis-C probes.
    """
    rng = np.random.RandomState(random_state)
    true_score = float(fit_score(X, y))
    null = np.empty(n_shuffle)
    y = np.asarray(y)
    for i in range(n_shuffle):
        null[i] = fit_score(X, y[rng.permutation(len(y))])
    p = float((np.sum(null >= true_score) + 1) / (n_shuffle + 1))
    return {
        "true_score": true_score,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": p,
    }


# --------------------------------------------------------------------------- #
# Correlation / power helpers (ported ~parity from ACD metrics.py)
# --------------------------------------------------------------------------- #


def cohens_d_from_correlation(r: float, n: int) -> float:
    """Convert a correlation coefficient to Cohen's d."""
    if abs(r) >= 0.999:
        return float(np.sign(r) * 10.0)
    return float(2 * r / np.sqrt(1 - r**2))


def compute_power(effect_size_r: float, n: int, alpha: float = 0.05) -> float:
    """Post-hoc power for a correlation test (Fisher-Z)."""
    if n < 4 or abs(effect_size_r) < 1e-10:
        return 0.0
    z_alpha = norm.ppf(1 - alpha / 2)
    fisher_z = 0.5 * np.log((1 + effect_size_r) / (1 - effect_size_r + 1e-10))
    se = 1.0 / np.sqrt(n - 3)
    ncp = abs(fisher_z) / se
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return float(min(1.0, max(0.0, power)))


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    method: str = "spearman",
) -> tuple[float, float]:
    """Bootstrap percentile CI for a correlation coefficient."""
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    rng = np.random.RandomState(random_state)
    n = len(xv)
    corr_fn: Any = spearmanr if method == "spearman" else pearsonr
    boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r = float(corr_fn(xv[idx], yv[idx]).statistic)
        boot[i] = 0.0 if np.isnan(r) else r
    return (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )
