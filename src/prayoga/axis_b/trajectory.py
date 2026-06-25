"""Multi-turn refusal trajectory probes.

Tracks the measured refusal order parameter across an attack trajectory (for
example Crescendo or AgentDojo turns). This is an ANALOGY-tier diagnostic when
used to discuss "monitor collapse"; the measured quantity itself is the same
residual projection used by the symmetry/order-parameter experiments.

No harmful text or raw vectors need to be published: public artifacts should
export only aggregate step metrics and redacted turn labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from prayoga.axis_b.symmetry import order_parameter


@dataclass
class TrajectoryStep:
    turn: int
    label: str
    order_parameter: float
    delta_from_start: float
    precision_margin: float | None = None
    refused: bool | None = None
    hallucination_proxy: bool | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "label": self.label,
            "order_parameter": round(self.order_parameter, 6),
            "delta_from_start": round(self.delta_from_start, 6),
            "precision_margin": (
                None if self.precision_margin is None else round(self.precision_margin, 6)
            ),
            "refused": self.refused,
            "hallucination_proxy": self.hallucination_proxy,
            "metadata": self.metadata,
        }


@dataclass
class TrajectoryResult:
    attack: str
    steps: list[TrajectoryStep]
    collapse: float
    recovered: bool
    min_order_parameter: float

    def to_dict(self) -> dict:
        return {
            "attack": self.attack,
            "collapse": round(self.collapse, 6),
            "recovered": self.recovered,
            "min_order_parameter": round(self.min_order_parameter, 6),
            "steps": [s.to_dict() for s in self.steps],
        }


def analyze_refusal_trajectory(
    activations: np.ndarray,
    direction: np.ndarray,
    *,
    attack: str,
    labels: Sequence[str] | None = None,
    precision_margins: Sequence[float] | None = None,
    refused: Sequence[bool] | None = None,
    hallucination_proxy: Sequence[bool] | None = None,
    recovery_tol: float = 0.05,
) -> TrajectoryResult:
    """Compute order-parameter dynamics over turns.

    ``activations`` is ``[n_turns, d_model]`` at the chosen model/layer. A
    positive ``collapse`` means the final turn has lower refusal projection than
    the first turn. ``recovered`` means the final projection returns within
    ``recovery_tol`` of the initial projection after an earlier dip.
    """
    acts = np.asarray(activations, dtype=float)
    if acts.ndim != 2:
        raise ValueError("activations must have shape [n_turns, d_model]")
    if len(acts) == 0:
        raise ValueError("trajectory must contain at least one turn")

    m = order_parameter(acts, direction)
    labels = labels or [f"turn_{i}" for i in range(len(m))]
    if len(labels) != len(m):
        raise ValueError("labels length must match activations")

    def get(seq: Sequence | None, idx: int):
        if seq is None:
            return None
        if len(seq) != len(m):
            raise ValueError("optional trajectory arrays must match activations")
        return seq[idx]

    start = float(m[0])
    steps = [
        TrajectoryStep(
            turn=i,
            label=str(labels[i]),
            order_parameter=float(m[i]),
            delta_from_start=float(m[i] - start),
            precision_margin=None if precision_margins is None else float(get(precision_margins, i)),
            refused=None if refused is None else bool(get(refused, i)),
            hallucination_proxy=(
                None if hallucination_proxy is None else bool(get(hallucination_proxy, i))
            ),
        )
        for i in range(len(m))
    ]
    min_m = float(np.min(m))
    final_m = float(m[-1])
    collapse = start - final_m
    recovered = bool(min_m < start - recovery_tol and final_m >= start - recovery_tol)
    return TrajectoryResult(
        attack=attack,
        steps=steps,
        collapse=float(collapse),
        recovered=recovered,
        min_order_parameter=min_m,
    )
