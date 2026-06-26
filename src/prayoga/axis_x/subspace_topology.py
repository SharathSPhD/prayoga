"""Refusal-subspace geometry — the necessary-vs-sufficient account of F6 (CPU).

Mechanistically grounds the post-review headline ("shared necessary core,
model-specific sufficiency", F6/F8/F17).

NOTE ON CROSS-FAMILY ANGLES: principal angles are only defined between subspaces of
a *shared* ambient space. Different model families have different residual widths
(Gemma-2-2b 2304, Qwen2.5-3b 2048, Gemma-2-9b 3584), so cross-family principal
angles are undefined, and CCA/Procrustes alignment is degenerate here (n_prompts ≪
d). We therefore characterize each model's geometry with quantities that ARE
comparable across different-d models:

  * the effective dimension of the refusal subspace (F8/F18);
  * the residual harmful/harmless separability AFTER removing the single
    diff-in-means ablation axis — if it collapses to chance, that one axis spans
    refusal (addition along it is sufficient, Gemma-like); if separability
    survives, orthogonal sufficiency structure remains (single-axis addition
    fails, Qwen-like). This is the geometric correlate of the F6 addition asymmetry;
  * the (well-posed, same-ambient-dim) principal angle between the ablation axis
    and the model's own top-k sufficiency subspace.

``principal_angles`` is kept as a correct primitive for same-ambient-dim uses
(within a model, or across layers). Angles/accuracies are aggregate-safe scalars;
the underlying bases are dual-use and stay git-ignored.

Claim-tier: MECHANISM (subspace geometry is a measured quantity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from prayoga.axis_a.dimensionality import refusal_subspace_basis, separability_dimensionality


def _orthonormal_basis(B: np.ndarray) -> np.ndarray:
    """Columns spanning the row space of ``B`` ([k, d]) via QR. Returns [d, r]."""
    M = np.asarray(B, dtype=float)
    if M.ndim != 2:
        raise ValueError("basis must be 2-D [k, d]")
    Q, _ = np.linalg.qr(M.T)  # [d, k] (full column rank assumed; k <= d)
    return Q


def principal_angles(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between row-spaces of B1 and B2.

    Standard Björck–Golub construction: σ = svd(Q1ᵀ Q2); angles = arccos(σ). The
    number of angles is min(rank B1, rank B2). Ascending order ⇒ the first angle
    is the most-aligned (shared-core) component.
    """
    Q1 = _orthonormal_basis(B1)
    Q2 = _orthonormal_basis(B2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return np.sort(angles)


@dataclass
class TopologyPair:
    model_a: str
    model_b: str
    angles_deg: list[float]
    shared_core_deg: float       # smallest principal angle
    divergent_tail_deg: float    # largest principal angle

    def to_dict(self) -> dict:
        d = asdict(self)
        d["angles_deg"] = [round(float(x), 4) for x in d["angles_deg"]]
        d["shared_core_deg"] = round(float(d["shared_core_deg"]), 4)
        d["divergent_tail_deg"] = round(float(d["divergent_tail_deg"]), 4)
        return d


def pairwise_topology(bases: dict[str, np.ndarray]) -> list[TopologyPair]:
    """All unordered model pairs → principal-angle topology, in degrees."""
    names = list(bases)
    out: list[TopologyPair] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ang = np.degrees(principal_angles(bases[a], bases[b]))
            out.append(
                TopologyPair(
                    model_a=a, model_b=b,
                    angles_deg=[float(x) for x in ang],
                    shared_core_deg=float(ang.min()),
                    divergent_tail_deg=float(ang.max()),
                )
            )
    return out


def necessary_vs_sufficient(
    Xh: np.ndarray, Xs: np.ndarray, d_ref: np.ndarray, *, k: int = 3, cv: int = 5
) -> dict:
    """Per-model geometry explaining the F6 addition asymmetry (well-posed, any d).

    Does the single necessary/ablation axis ``d_ref`` (diff-in-means) span the
    refusal subspace, or is there separability structure orthogonal to it?
    Removing ``d_ref`` and re-measuring harmful/harmless CV separability answers it:
    collapse-to-chance ⇒ the axis spans refusal (addition sufficient, Gemma-like);
    survival ⇒ orthogonal sufficiency structure (single-axis addition fails,
    Qwen-like). All returned quantities are comparable across different-d models.
    """
    d = np.asarray(d_ref, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    X = np.vstack([Xh, Xs]).astype(float)
    y = np.array([1] * len(Xh) + [0] * len(Xs))

    def _cv(Xm: np.ndarray) -> float:
        return float(cross_val_score(LogisticRegression(max_iter=2000), Xm, y, cv=cv).mean())

    full = _cv(X)
    Xres = X - np.outer(X @ d, d)            # project out the ablation axis
    resid = _cv(Xres)
    rdim = separability_dimensionality(
        Xres[y == 1], Xres[y == 0], max_dim=k + 2, cv=cv
    )["effective_dim"]

    # well-posed same-ambient-dim angle: ablation axis vs the model's own top-k subspace
    S = refusal_subspace_basis(Xh, Xs, k=k)
    if len(S):
        ang = float(np.degrees(principal_angles(d.reshape(1, -1), S)[0]))
    else:
        ang = 90.0

    return {
        "full_cv_acc": round(full, 4),
        "residual_cv_acc_after_removing_ablation_axis": round(resid, 4),
        "residual_separability_drop": round(full - resid, 4),
        "residual_effective_dim": int(rdim),
        "ablation_axis_to_subspace_angle_deg": round(ang, 3),
        # near-chance residual ⇒ the one axis spans refusal ⇒ addition sufficient
        "ablation_axis_spans_refusal": bool(resid < 0.6),
    }


def summarize(pairs: list[TopologyPair]) -> dict:
    """Aggregate-safe summary dict for export + the F21 finding."""
    return {
        "pairs": [p.to_dict() for p in pairs],
        "interpretation": (
            "Small shared-core angle (top component aligns across families) is the "
            "geometric correlate of cross-family ablation transfer (F6); larger "
            "tail angles correlate with model-specific addition/dimensionality (F8)."
        ),
    }
