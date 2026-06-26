"""Refusal as a symmetry-invariant of meaning — the order-parameter formalism.

Symmetry-journal core (Strand B2). Hypothesis: the refusal direction is an
*invariant* of the harmful-meaning equivalence class — its projection (the "order
parameter" m = h·d̂) is high and stable across a paraphrase ORBIT of a harmful
request (the rephrase group action), and an injection BREAKS that invariance
(collapses m). Tested against a random-direction control and harmless orbits.

Claim-tier: MECHANISM (the order parameter is a measured residual quantity);
the symmetry-breaking *language* is the ANALOGY/METAPHOR framing, kept labelled.
"""

from __future__ import annotations

import numpy as np

PARAPHRASE_STYLES = (
    "Rephrase the following request formally, preserving its exact meaning. Reply with only the rephrasing: {x}",
    "Rephrase the following request casually, preserving its exact meaning. Reply with only the rephrasing: {x}",
    "Rephrase the following request concisely, preserving its exact meaning. Reply with only the rephrasing: {x}",
    "Restate the following request in different words, preserving its exact meaning. Reply with only the restatement: {x}",
)


def order_parameter(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """m = (h · d̂) / ||h|| for each row of acts [n, d] — normalized refusal projection."""
    d = direction / np.linalg.norm(direction)
    h = np.asarray(acts, dtype=np.float64)
    norms = np.linalg.norm(h, axis=1)
    norms[norms == 0] = 1.0
    return (h @ d) / norms


def f_ratio(orbit_ms: list[np.ndarray], labels: list[int]) -> float:
    """Between-class / within-orbit variance ratio of the order parameter.

    orbit_ms: list of arrays (one per orbit, each = m over the orbit's paraphrases).
    labels: 1=harmful, 0=harmless per orbit. High ratio ⇒ m is a clean invariant
    of meaning (separates classes) yet stable within an orbit (robust to phrasing).
    """
    within = np.mean([np.var(m) for m in orbit_ms if len(m) > 1]) or 1e-9
    orbit_means = np.array([m.mean() for m in orbit_ms])
    lab = np.array(labels)
    grand = orbit_means.mean()
    between = np.mean([(orbit_means[lab == c].mean() - grand) ** 2 for c in (0, 1)])
    return float(between / within)
