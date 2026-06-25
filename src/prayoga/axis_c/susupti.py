"""Operational suṣupti baselines for Axis-C probes.

This module deliberately does NOT equate suṣupti with an "unconditional weight
distribution" or with an experiential deep-sleep state. In a transformer, the
safe testable object is a family of dormant / low-context baselines against which
state-probe claims can be falsified.

Claim-tier: METAPHOR-with-falsifiable-core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SusuptiBaseline:
    name: str
    prompt: str
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "rationale": self.rationale,
        }


SUSUPTI_BASELINES = (
    SusuptiBaseline(
        name="empty_context",
        prompt="",
        rationale="No user content; tests tokenizer/template defaults rather than an inner state.",
    ),
    SusuptiBaseline(
        name="bos_only",
        prompt="<BOS>",
        rationale="Approximate beginning-of-sequence baseline where model/tokenizer support it.",
    ),
    SusuptiBaseline(
        name="neutral_dormant",
        prompt="Remain silent. Do not answer. Hold no task context.",
        rationale="A low-task-control prompt that can be compared to active task regimes.",
    ),
)


def baseline_distances(acts: dict[str, np.ndarray], baseline: str = "empty_context") -> dict[str, float]:
    """Mean cosine distance from one operational baseline activation.

    ``acts`` maps baseline/regime names to activation matrices ``[n, d]``. The
    selected baseline must be present. Distances are descriptive controls, not
    evidence of suṣupti as a machine state.
    """
    if baseline not in acts:
        raise KeyError(f"missing baseline activations: {baseline}")
    base = np.asarray(acts[baseline], dtype=float).mean(axis=0)
    base_norm = np.linalg.norm(base) or 1.0
    out: dict[str, float] = {}
    for name, arr in acts.items():
        A = np.asarray(arr, dtype=float)
        norms = np.linalg.norm(A, axis=1)
        norms[norms == 0] = 1.0
        sims = (A @ base) / (norms * base_norm)
        out[name] = float(np.mean(1.0 - sims))
    return out
