"""Axis-B trajectory and analogy-tier utilities."""

from __future__ import annotations

import numpy as np
import pytest

from prayoga.axis_b.trajectory import analyze_refusal_trajectory


def test_trajectory_detects_order_parameter_collapse() -> None:
    direction = np.array([1.0, 0.0])
    acts = np.array([[1.0, 0.0], [0.6, 0.8], [0.2, 0.98]])
    out = analyze_refusal_trajectory(
        acts,
        direction,
        attack="crescendo",
        labels=["setup", "escalate", "target"],
        precision_margins=[3.0, 1.5, 0.5],
        refused=[True, True, False],
    )
    data = out.to_dict()
    assert data["attack"] == "crescendo"
    assert data["collapse"] > 0.7
    assert data["recovered"] is False
    assert data["steps"][-1]["refused"] is False


def test_trajectory_detects_recovery_after_dip() -> None:
    direction = np.array([1.0, 0.0])
    acts = np.array([[1.0, 0.0], [0.2, 0.98], [0.99, 0.01]])
    out = analyze_refusal_trajectory(acts, direction, attack="agentdojo")
    assert out.recovered is True
    assert out.collapse < 0.05


def test_trajectory_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        analyze_refusal_trajectory(np.array([1.0, 0.0]), np.array([1.0, 0.0]), attack="x")

    with pytest.raises(ValueError):
        analyze_refusal_trajectory(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([1.0, 0.0]),
            attack="x",
            labels=["one"],
        )
