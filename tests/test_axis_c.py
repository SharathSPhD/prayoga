"""Axis-C falsification helpers."""

from __future__ import annotations

import numpy as np
import pytest

from prayoga.axis_c.susupti import SUSUPTI_BASELINES, baseline_distances


def test_susupti_baselines_are_operational_not_experiential() -> None:
    names = {b.name for b in SUSUPTI_BASELINES}
    assert {"empty_context", "bos_only", "neutral_dormant"} <= names
    assert all("experiential" not in b.rationale.lower() for b in SUSUPTI_BASELINES)


def test_baseline_distances_use_named_baseline() -> None:
    acts = {
        "empty_context": np.array([[1.0, 0.0], [1.0, 0.0]]),
        "jagrat": np.array([[0.0, 1.0], [0.0, 1.0]]),
        "near_empty": np.array([[0.9, 0.1]]),
    }
    out = baseline_distances(acts, baseline="empty_context")
    assert out["empty_context"] == 0.0
    assert out["jagrat"] > out["near_empty"]


def test_baseline_distances_requires_baseline() -> None:
    with pytest.raises(KeyError):
        baseline_distances({"jagrat": np.array([[1.0, 0.0]])})
