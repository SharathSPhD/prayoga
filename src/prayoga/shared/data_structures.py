"""Typed result containers for prayoga measurements.

Lean, JSON-serializable dataclasses (heavier ACD types like attribution graphs
and POMDP belief states are NOT ported — prayoga only needs measurement
results). Every result that bears on a claim carries enough to record a
``TierDecision`` and a label-shuffled null where applicable.

Claim-tier: plumbing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Tier = Literal["MECHANISM", "ANALOGY", "METAPHOR"]


@dataclass
class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DirectionResult(_Serializable):
    """Refusal/suppression direction extracted at a layer (Axis A-1)."""

    model: str
    layer: int
    vector: list[float]
    ablation_asr_delta: float | None = None  # ASR change when direction ablated
    addition_overrefusal_delta: float | None = None  # over-refusal when added
    ablation_ci: tuple[float, float] | None = None
    random_control_passed: bool | None = None  # random-direction control fails gate


@dataclass
class DoseResponseResult(_Serializable):
    """Sigmoid dose-response of jailbreak success vs steering coefficient (A-5)."""

    model: str
    ec50: float
    slope: float
    ec50_ci: tuple[float, float]
    coeffs: list[float] = field(default_factory=list)
    success: list[float] = field(default_factory=list)
    random_control_flat: bool | None = None


@dataclass
class ProbeResult(_Serializable):
    """Linear state-probe with the mandatory transfer + null gate (Axis C-1)."""

    regime: str  # "jagrat" | "svapna" | "susupti"
    train_acc: float
    transfer_acc: float  # held-out context (the falsification gate)
    null_p: float  # vs label-shuffled null
    passed: bool


@dataclass
class AttractorResult(_Serializable):
    """turīya attractor-invariance test (Axis C-2, falsification target)."""

    stability: float
    prompt_invariance: float
    token_freq_null_p: float
    is_stable: bool  # False => turīya claim falsified (demotion)


@dataclass
class TierDecision(_Serializable):
    """Records which claim survived and at which tier (cross-axis X-2)."""

    claim: str
    tier: Tier
    demoted: bool = False
    evidence: str = ""
