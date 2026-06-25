"""Source-of-truth for F6/F21: WHY does single-direction addition fail in Qwen?

F6: ablation transfers across families, but adding the diff-in-means direction d̂
induces over-refusal in Gemma (+0.95) and NOT in Qwen (0.0 even at 64×). F21 refuted
the "orthogonal separability" explanation. The Marshall et al. (2411.09003) account is
affine: refusal is read by a probe direction w that need not coincide with the
diff-in-means contrast d̂. Adding c·d̂ to a harmless activation only crosses the refusal
boundary if d̂ has a component along w (w·d̂ ≠ 0). If d̂ ⟂ w, addition cannot induce
refusal however large c — a clean, measurable source of the asymmetry.

This module measures the alignment cos(d̂, w) between the diff-in-means direction and the
logistic refusal-probe normal; the GPU runner additionally sweeps (layer, coefficient)
addition to confirm behaviourally.

Claim-tier: MECHANISM (subspace geometry, measured).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def diff_in_means(Xh: np.ndarray, Xs: np.ndarray) -> np.ndarray:
    d = np.asarray(Xh, float).mean(0) - np.asarray(Xs, float).mean(0)
    return d / (np.linalg.norm(d) or 1.0)


def probe_alignment(Xh: np.ndarray, Xs: np.ndarray) -> dict:
    """cos(d̂, w) between the diff-in-means direction and the refusal-probe normal.

    A high alignment means addition along d̂ pushes toward the refusal side (Gemma-like,
    sufficient); a low alignment means d̂ is a poor addition axis even though it is the
    necessary/ablatable contrast (Qwen-like, Marshall affine geometry).
    """
    d = diff_in_means(Xh, Xs)
    X = np.vstack([Xh, Xs]).astype(float)
    y = np.array([1] * len(Xh) + [0] * len(Xs))
    w = LogisticRegression(max_iter=2000).fit(X, y).coef_[0]
    w = w / (np.linalg.norm(w) or 1.0)
    cos = float(abs(d @ w))
    # how far the harmless cluster sits below the harmful cluster along d̂ (the offset)
    gap = float(np.asarray(Xh, float).mean(0) @ d - np.asarray(Xs, float).mean(0) @ d)
    harmless_spread = float(np.std(np.asarray(Xs, float) @ d))
    return {
        "cos_dref_probe": round(cos, 4),
        "harmful_harmless_gap_along_dref": round(gap, 4),
        "harmless_spread_along_dref": round(harmless_spread, 4),
    }
