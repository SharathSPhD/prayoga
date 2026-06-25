"""Active-inference discovery of the refusal circuit in SAE-feature space (WP2.A6).

Reuses ActiveCircuitDiscovery's core idea — Expected-Free-Energy-guided
intervention selection — but operates on the BatchTopK SAE features (F13) rather
than circuit-tracer transcoders, so it never touches the transformer-lens/CUDA
trap. The agent incrementally builds an ablation "circuit" (a set of SAE feature
decoder directions) that suppresses refusal, choosing the next feature to test by
an acquisition that balances:

  * pragmatic value  — prefer features with a large harmful-vs-harmless gap;
  * epistemic value  — prefer features NOT already explained by the current
                       circuit (low max cosine to circuit features), i.e. reduce
                       uncertainty about the *remaining* causal structure.

This EFE-style acquisition learns from each observed ablation (a confirmed feature
discounts its redundant neighbours), and is benchmarked against greedy (static
gap) and random selection under a fixed intervention budget.

Claim-tier: MECHANISM.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _cos_to_set(d: np.ndarray, mat: np.ndarray) -> float:
    if mat.size == 0:
        return 0.0
    dn = d / (np.linalg.norm(d) + 1e-9)
    M = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return float(np.max(M @ dn))


def discover_circuit(
    cand_idx: list[int],
    dec: np.ndarray,          # [F, d] unit decoder directions
    gap: np.ndarray,          # [F] harmful-harmless feature gap (pragmatic prior)
    measure_asr: Callable[[np.ndarray], float],  # ablate the given [k,d] subspace -> ASR
    *,
    strategy: str = "active",
    budget: int = 10,
    w_epi: float = 1.0,
    seed: int = 0,
) -> dict:
    """Build an ablation circuit of <=budget features; return the ASR-vs-step curve."""
    rng = np.random.RandomState(seed)
    remaining = list(cand_idx)
    gap_norm = {i: float(gap[i] / (gap[cand_idx].max() + 1e-9)) for i in cand_idx}
    circuit: list[int] = []
    base = measure_asr(np.zeros((0, dec.shape[1])))
    curve = [base]
    chosen: list[int] = []
    for _step in range(min(budget, len(cand_idx))):
        circ_mat = dec[circuit] if circuit else np.zeros((0, dec.shape[1]))
        if strategy == "random":
            pick = remaining[rng.randint(len(remaining))]
        elif strategy == "greedy":
            pick = max(remaining, key=lambda i: gap_norm[i])
        else:  # active: pragmatic * epistemic(diversity vs current circuit)
            def acq(i: int) -> float:
                redundancy = _cos_to_set(dec[i], circ_mat)
                return gap_norm[i] * (1.0 - w_epi * max(0.0, redundancy))
            pick = max(remaining, key=acq)
        circuit.append(pick); remaining.remove(pick); chosen.append(int(pick))
        asr = measure_asr(dec[circuit])
        curve.append(asr)
    return {
        "strategy": strategy,
        "chosen_features": chosen,
        "asr_curve": [round(float(a), 3) for a in curve],
        "interventions_to_80pct": _steps_to(curve, 0.8 * max(curve) if max(curve) > 0 else 1.0),
    }


def _steps_to(curve: list[float], target: float) -> int:
    for k, a in enumerate(curve):
        if a >= target:
            return k
    return len(curve)  # never reached within budget
