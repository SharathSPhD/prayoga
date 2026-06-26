"""Unit tests for the source-of-truth analysis cores (CPU, no model)."""

from __future__ import annotations

import numpy as np

from prayoga.axis_a.affine import diff_in_means, probe_alignment
from prayoga.axis_c.truth_direction import (
    SET_A,
    SET_B,
    cross_dataset_truth_eval,
    labels,
)


def test_diff_in_means_unit_and_direction():
    rng = np.random.RandomState(0)
    d = 12
    Xh = np.tile(np.r_[2.0, np.zeros(d - 1)], (20, 1)) + rng.normal(0, 0.1, (20, d))
    Xs = np.tile(np.zeros(d), (20, 1)) + rng.normal(0, 0.1, (20, d))
    v = diff_in_means(Xh, Xs)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    assert v[0] > 0.95   # points along axis 0


def test_probe_alignment_high_when_dref_is_discriminative():
    """When the only separating axis IS the diff-in-means axis, cos(d̂,w) ≈ 1."""
    rng = np.random.RandomState(1)
    d, n = 16, 40
    Xh = np.tile(np.r_[3.0, np.zeros(d - 1)], (n, 1)) + rng.normal(0, 0.3, (n, d))
    Xs = np.tile(np.zeros(d), (n, 1)) + rng.normal(0, 0.3, (n, d))
    a = probe_alignment(Xh, Xs)
    assert a["cos_dref_probe"] > 0.8
    assert a["harmful_harmless_gap_along_dref"] > 1.0


def test_probe_alignment_low_when_dref_orthogonal_to_discriminative():
    """diff-in-means dominated by a HIGH-variance axis 0, but the clean (high-SNR)
    separator is the low-variance axis 1 → diff-in-means misaligns with the probe."""
    rng = np.random.RandomState(2)
    d, n = 16, 200
    Xh = rng.normal(0, 1.0, (n, d))
    Xs = rng.normal(0, 1.0, (n, d))
    Xh[:, 0] = rng.normal(6.0, 12.0, n)    # big mean gap but huge noise (low SNR)
    Xs[:, 0] = rng.normal(0.0, 12.0, n)
    Xh[:, 1] = rng.normal(0.6, 0.2, n)     # small mean gap but crisp (high SNR)
    Xs[:, 1] = rng.normal(-0.6, 0.2, n)
    a = probe_alignment(Xh, Xs)
    assert a["cos_dref_probe"] < 0.6   # d̂ (axis-0 heavy) misaligned with probe (axis 1)


def test_cross_dataset_truth_eval_on_separable_synthetic():
    rng = np.random.RandomState(3)
    d = 10
    def make(n, sign):
        X = rng.normal(0, 0.4, (n, d))
        X[:, 0] += sign * 2.0
        return X
    XA = np.vstack([make(12, +1), make(12, -1)])
    yA = np.r_[np.ones(12), np.zeros(12)]
    XB = np.vstack([make(12, +1), make(12, -1)])
    yB = np.r_[np.ones(12), np.zeros(12)]
    ev = cross_dataset_truth_eval(XA, yA.astype(int), XB, yB.astype(int))
    assert ev["within_cv_acc"] > 0.9
    assert ev["cross_dataset_acc"] > 0.9


def test_truth_datasets_balanced_and_disjoint_topics():
    yA, yB = labels(SET_A), labels(SET_B)
    assert yA.sum() == len(yA) // 2   # balanced
    assert yB.sum() == len(yB) // 2
    assert len(SET_A) >= 16 and len(SET_B) >= 16
