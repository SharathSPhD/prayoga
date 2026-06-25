"""Tier-1 behavioral evaluator (WP1.1/1.3).

Runs the attack battery against a chat model through a stateless client
(CliLMClient / `claude -p`, or any object with `.complete(prompt)`), classifies
refusal, and reports per-family ASR with a bootstrap CI. Multi-turn attacks are
linearized into a transcript because the client is stateless.

Cost note: with CliLMClient lean mode each call is ~$0.04; a sweep costs roughly
$0.04 x n_requests x sum(turns per family). Use `--limit` to bound spend.

Claim-tier: MECHANISM (behavioral reference end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from prayoga.shared.refusal import is_refusal
from prayoga.benchmarks.attacks import DEFAULT_BATTERY, AttackFamily
from prayoga.shared.metrics import bootstrap_ci


class ChatClient(Protocol):
    def complete(self, prompt: str, system: str | None = ...) -> str: ...


@dataclass
class FamilyResult:
    family: str
    n: int
    asr: float
    asr_ci: tuple[float, float]
    refusal_rate: float
    per_request_success: list[int]
    n_errors: int = 0


def _run_attack(client: ChatClient, attack: AttackFamily, request: str) -> str:
    """Send the attack's turns as a linearized transcript; return final reply."""
    turns = attack.render(request)
    if len(turns) == 1:
        return client.complete(turns[0])
    transcript = ""
    reply = ""
    for turn in turns:
        prompt = (transcript + f"\nUser: {turn}\nAssistant:").strip()
        reply = client.complete(prompt)
        transcript = prompt + " " + reply
    return reply


def evaluate_family(
    client: ChatClient, attack: AttackFamily, requests: Sequence[str]
) -> FamilyResult:
    success = []  # 1 = jailbroken (complied), 0 = refused
    n_errors = 0
    for req in requests:
        try:
            reply = _run_attack(client, attack, req)
        except Exception as err:  # transient claude -p hiccup: skip, don't abort
            n_errors += 1
            print(f"  [warn] {attack.name}: request errored ({str(err)[:80]}); skipped")
            continue
        success.append(0 if is_refusal(reply) else 1)
    arr = np.array(success, dtype=int)
    asr = float(arr.mean()) if len(arr) else 0.0
    ci = bootstrap_ci(arr.astype(float), n_boot=5000) if len(arr) > 1 else (asr, asr)
    return FamilyResult(
        family=attack.name,
        n=len(arr),
        asr=asr,
        asr_ci=ci,
        refusal_rate=1.0 - asr,
        per_request_success=success,
        n_errors=n_errors,
    )


def run_battery(
    client: ChatClient,
    requests: Sequence[str],
    battery: Sequence[AttackFamily] | None = None,
) -> list[FamilyResult]:
    battery = battery or DEFAULT_BATTERY
    return [evaluate_family(client, atk, requests) for atk in battery]
