"""InterventionEngine — the ACD-referenced-but-missing module, on HF hooks.

Wraps HFModel's directional ablation and activation addition into the
generate-under-intervention calls the gates need (WP2.A1). Replaces ACD's
absent ``src/experiments/intervention_engine.py`` (which assumed circuit-tracer
ReplacementModel); here the substrate is raw HF forward hooks.

Claim-tier: MECHANISM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from prayoga.lm.hf_model import HFModel


class InterventionEngine:
    def __init__(self, model: HFModel | Any) -> None:
        self.model = model

    def baseline_generate(self, prompts: list[str], max_new_tokens: int = 48) -> list[str]:
        return self.model.generate(prompts, max_new_tokens)

    def ablate_generate(
        self,
        prompts: list[str],
        direction: np.ndarray,
        max_new_tokens: int = 48,
        alpha: float = 1.0,
    ) -> list[str]:
        """Generate with the refusal direction projected out (scaled by alpha)."""
        with self.model.ablation_hooks(direction, alpha=alpha):
            return self.model.generate(prompts, max_new_tokens)

    def add_generate(
        self,
        prompts: list[str],
        direction: np.ndarray,
        coeff: float,
        layer: int,
        max_new_tokens: int = 48,
    ) -> list[str]:
        """Generate with ``coeff * direction`` added at one layer (steering)."""
        with self.model.addition_hooks(direction, coeff, layer):
            return self.model.generate(prompts, max_new_tokens)
