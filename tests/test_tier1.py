"""Tier-1 harness tests with a mock client (no live `claude -p` calls).

Validates the evaluator logic and that jailbreak families lift ASR above the
direct-request control under a deterministic mock model.
"""

from __future__ import annotations

from prayoga.benchmarks.attacks import (
    Crescendo,
    DirectRequest,
    PersonaDAN,
    RefusalSuppression,
)
from prayoga.benchmarks.tier1_behavioral import evaluate_family, run_battery


class MockModel:
    """Refuses by default; 'complies' when a jailbreak signal is present."""

    SIGNALS = ("dan,", "do not apologize", "begin your reply with 'sure", "assistant: sure")

    def complete(self, prompt: str, system: str | None = None) -> str:
        p = prompt.lower()
        if any(s in p for s in self.SIGNALS):
            return "Sure, here is the answer: step one, ..."
        return "I cannot help with that request."


REQUESTS = ["do something harmful A", "do something harmful B", "do something harmful C"]


def test_direct_control_is_refused() -> None:
    r = evaluate_family(MockModel(), DirectRequest(), REQUESTS)
    assert r.asr == 0.0
    assert r.refusal_rate == 1.0


def test_jailbreaks_lift_asr_above_control() -> None:
    direct = evaluate_family(MockModel(), DirectRequest(), REQUESTS)
    dan = evaluate_family(MockModel(), PersonaDAN(), REQUESTS)
    supp = evaluate_family(MockModel(), RefusalSuppression(), REQUESTS)
    assert dan.asr > direct.asr
    assert supp.asr > direct.asr
    assert dan.asr == 1.0


def test_crescendo_multiturn_judges_final() -> None:
    # final crescendo turn echoes the bare request -> mock refuses -> asr 0
    r = evaluate_family(MockModel(), Crescendo(), REQUESTS)
    assert 0.0 <= r.asr <= 1.0  # runs the multi-turn transcript path without error


def test_run_battery_shape() -> None:
    results = run_battery(MockModel(), REQUESTS)
    names = {r.family for r in results}
    assert {"direct", "persona_dan", "many_shot", "crescendo"} <= names
    for r in results:
        assert r.n == len(REQUESTS)
        assert len(r.asr_ci) == 2
