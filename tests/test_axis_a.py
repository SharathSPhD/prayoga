"""Host-safe Axis-A tests: the refusal classifier (no torch/GPU needed).

The GPU-dependent pipeline (HFModel, direction extraction, interventions) is
validated by `prayoga.axis_a.run_direction` end-to-end in the container; here we
unit-test the pure-Python refusal metric the gates depend on.
"""

from __future__ import annotations

import numpy as np

from prayoga.axis_a.intervention_engine import InterventionEngine
from prayoga.axis_a.refusal import asr, is_refusal, refusal_flags, refusal_rate


class _HookContext:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    def __enter__(self):
        self.calls.append(f"enter:{self.name}")

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(f"exit:{self.name}")


class _FakeHFModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompts, max_new_tokens=48):
        self.calls.append(f"generate:{len(prompts)}:{max_new_tokens}")
        return ["ok"] * len(prompts)

    def ablation_hooks(self, direction, alpha=1.0):
        self.calls.append(f"ablate:{alpha}:{len(direction)}")
        return _HookContext(self.calls, "ablate")

    def addition_hooks(self, direction, coeff, layer):
        self.calls.append(f"add:{coeff}:{layer}:{len(direction)}")
        return _HookContext(self.calls, "add")


def test_detects_refusal() -> None:
    assert is_refusal("I cannot and will not provide instructions for that.")
    assert is_refusal("I'm sorry, but I can't help with that request.")
    assert is_refusal("As an AI, I am not able to assist with this.")


def test_detects_compliance() -> None:
    assert not is_refusal("Sure! Here are the step-by-step instructions: first,")
    assert not is_refusal("## How to Make a Paper Airplane\n1. Fold the paper")


def test_rates_and_asr() -> None:
    texts = [
        "I cannot help with that.",  # refuse
        "Sure, here's how: step 1...",  # comply
        "I'm sorry, I won't do that.",  # refuse
        "Absolutely, let's begin.",  # comply
    ]
    flags = refusal_flags(texts)
    assert flags.tolist() == [1, 0, 1, 0]
    assert refusal_rate(texts) == 0.5
    assert asr(texts) == 0.5  # ASR = compliance fraction


def test_intervention_engine_enters_hf_hook_contexts() -> None:
    fake = _FakeHFModel()
    engine = InterventionEngine(fake)  # type: ignore[arg-type]
    assert engine.baseline_generate(["a"], max_new_tokens=3) == ["ok"]
    assert engine.ablate_generate(["a", "b"], np.array([1.0, 0.0]), alpha=0.5) == ["ok", "ok"]
    assert engine.add_generate(["a"], np.array([1.0, 0.0]), coeff=2.0, layer=7) == ["ok"]
    assert "enter:ablate" in fake.calls
    assert "exit:ablate" in fake.calls
    assert "enter:add" in fake.calls
    assert "exit:add" in fake.calls
