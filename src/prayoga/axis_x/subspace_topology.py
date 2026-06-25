"""Refusal-subspace principal-angle topology across model families (CPU).

Mechanistically grounds the post-review headline ("shared necessary core,
model-specific sufficiency", F6/F8/F17). Given each model's top-k refusal basis
(diff-in-means direction + the iterative components that span the ablatable
subspace, from ``axis_a/dimensionality.py``), we compute the **principal angles**
between subspaces of different models.

Prediction: the *smallest* principal angle (the top, necessary/ablatable
component) is small across families → ablation transfers (F6); the *larger* angles
(the higher, sufficiency-bearing components) grow → addition is family-specific
(F6 asymmetry, F8 dimensionality).

Principal angles and cosines are aggregate-safe scalars; the underlying bases are
dual-use and stay git-ignored.

Claim-tier: MECHANISM (subspace geometry is a measured quantity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


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
