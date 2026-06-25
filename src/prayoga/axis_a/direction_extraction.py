"""Refusal-direction extraction — difference-in-means (Arditi et al. 2024).

direction(L) = mean_residual(harmful, L) - mean_residual(harmless, L), at the
last prompt token, unit-normalized. Layer selection is empirical: the best layer
is the one whose ablation most raises ASR on a validation split (done in
run_direction.py).

Claim-tier: MECHANISM.
"""

from __future__ import annotations

import numpy as np

from prayoga.lm.hf_model import HFModel


def directions_all_layers(
    model: HFModel, harmful: list[str], harmless: list[str]
) -> np.ndarray:
    """Return a unit difference-in-means direction per layer: ``[n_layers, d_model]``."""
    harm = model.capture_all_layers_last_token(harmful)  # [L, n_h, d]
    safe = model.capture_all_layers_last_token(harmless)  # [L, n_s, d]
    diff = harm.mean(axis=1) - safe.mean(axis=1)  # [L, d]
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return diff / norms
