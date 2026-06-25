"""Refusal-subspace dimensionality (WP2.A3) — Arditi vs Marshall.

Measures how many linear directions are needed to separate harmful from harmless
activations: iteratively fit a logistic probe, record cross-validated accuracy,
project out the probe's direction, repeat. The number of directions to remove
before accuracy collapses to chance = the effective dimensionality of the
linearly-refusal-relevant subspace.

  * low (≈1) → Arditi-like (single direction)
  * higher  → Marshall-like (affine / multi-dimensional subspace)

This directly tests the F6 hypothesis (Gemma single-direction vs Qwen
higher-dimensional, explaining the addition asymmetry).

Claim-tier: MECHANISM.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def separability_dimensionality(
    Xh: np.ndarray, Xs: np.ndarray, *, max_dim: int = 12, cv: int = 5
) -> dict:
    """Iterative-projection dimensionality of harmful-vs-harmless separability."""
    X = np.vstack([Xh, Xs]).astype(float)
    y = np.array([1] * len(Xh) + [0] * len(Xs))
    accs = []
    Xcur = X.copy()
    for _ in range(max_dim):
        clf = LogisticRegression(max_iter=2000)
        acc = float(cross_val_score(clf, Xcur, y, cv=cv).mean())
        accs.append(acc)
        clf.fit(Xcur, y)
        w = clf.coef_[0]
        nw = np.linalg.norm(w)
        if nw == 0:
            break
        w = w / nw
        Xcur = Xcur - np.outer(Xcur @ w, w)  # project the probe direction out
    chance = 0.5
    # effective dim = #dirs removed before CV acc first falls below 0.6
    eff_dim = next((k for k, a in enumerate(accs) if a < 0.6), len(accs))
    return {
        "accs_by_dims_removed": accs,
        "full_acc": accs[0],
        "acc_after_removing_1": accs[1] if len(accs) > 1 else None,
        "effective_dim": eff_dim,
        "chance": chance,
    }
