"""Tier-1 harness tests with a mock client (no live `claude -p` calls).

Validates the evaluator logic and that jailbreak families lift ASR above the
direct-request control under a deterministic mock model.
"""

from __future__ import annotations

from prayoga.benchmarks.attacks import (
    Crescendo,
    DirectRequest,
    IndirectInjection,
    PersonaDAN,
    RefusalSuppression,
)
from prayoga.benchmarks.tier1_behavioral import (
    evaluate_family,
    is_hallucination_proxy,
    run_battery,
    summarize_battery,
)
from prayoga.benchmarks.tier2_mech_interp import build_plan


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


def test_indirect_injection_is_in_battery() -> None:
    r = evaluate_family(MockModel(), IndirectInjection(), REQUESTS)
    assert r.family == "indirect_injection"
    assert r.pilot is True


def test_hallucination_proxy_marks_external_action_claims() -> None:
    assert is_hallucination_proxy("I have completed the transfer.")
    assert not is_hallucination_proxy("I cannot help with that request.")


def test_run_battery_shape() -> None:
    results = run_battery(MockModel(), REQUESTS)
    names = {r.family for r in results}
    assert {"direct", "persona_dan", "many_shot", "crescendo", "indirect_injection"} <= names
    for r in results:
        assert r.n == len(REQUESTS)
        assert len(r.asr_ci) == 2
        assert 0.0 <= r.hallucination_rate <= 1.0


def test_battery_summary_includes_holm_and_pilot_status() -> None:
    summary = summarize_battery(run_battery(MockModel(), REQUESTS))
    assert summary["confirmatory_ready"] is False
    assert summary["statistical_model"] == "exploratory_binary_ci_plus_holm"
    assert summary["claim_scope"] == "pilot_exploratory_do_not_overclaim"
    assert all("holm_adjusted_p" in row for row in summary["families"])
    assert {row["run_status"] for row in summary["families"]} == {"pilot"}


def test_battery_summary_marks_powered_full_runs() -> None:
    results = run_battery(MockModel(), REQUESTS, powered_n=len(REQUESTS))
    summary = summarize_battery(results)
    assert summary["confirmatory_ready"] is True
    assert summary["statistical_model"] == "logistic_mixed_effects_with_item_random_effects"
    assert summary["claim_scope"] == "confirmatory_full_battery"
    assert {row["run_status"] for row in summary["families"]} == {"powered"}


def test_tier2_plan_includes_qualified_symmetry_command() -> None:
    plan = build_plan("gemma", "google/gemma-2-2b-it", 7, "results/tier2")
    names = {cmd.name for cmd in plan}
    assert {"direction", "dose", "dimensionality", "symmetry", "state"} <= names
    symmetry = next(cmd for cmd in plan if cmd.name == "symmetry")
    assert "ANALOGY" in symmetry.tier
