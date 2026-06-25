"""Tier-2 mechanistic interpretability orchestrator.

This module provides a reproducible command plan for per-model Axis-A/B/C runs.
It does not execute GPU-heavy experiments by default; use ``--dry-run`` to print
the plan, or run selected commands manually inside the documented DGX container.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tier2Command:
    name: str
    module: str
    args: list[str]
    tier: str
    claim: str

    def argv(self) -> list[str]:
        return ["python", "-m", self.module, *self.args]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "args": self.args,
            "argv": self.argv(),
            "tier": self.tier,
            "claim": self.claim,
        }


def build_plan(model: str, hf_id: str, layer: int, out_dir: str) -> list[Tier2Command]:
    base = ["--model", model, "--hf-id", hf_id, "--layer", str(layer), "--out", out_dir]
    return [
        Tier2Command(
            name="direction",
            module="prayoga.axis_a.run_direction",
            args=base,
            tier="MECHANISM",
            claim="extract and causally test the refusal direction",
        ),
        Tier2Command(
            name="dose",
            module="prayoga.axis_a.run_dose",
            args=base,
            tier="MECHANISM",
            claim="fit refusal-suppression dose-response and EC50",
        ),
        Tier2Command(
            name="dimensionality",
            module="prayoga.axis_a.run_dimensionality",
            args=base,
            tier="MECHANISM",
            claim="decide extraction-layer effective refusal dimension",
        ),
        Tier2Command(
            name="symmetry",
            module="prayoga.axis_b.run_symmetry",
            args=base,
            tier="MECHANISM measurement / ANALOGY interpretation",
            claim="measure paraphrase-orbit order parameter and injection collapse",
        ),
        Tier2Command(
            name="state",
            module="prayoga.axis_c.run_state",
            args=["--model", model, "--hf-id", hf_id, "--out", out_dir],
            tier="METAPHOR",
            claim="attempt avasthātraya-style probe under demotion gates",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default="results/tier2")
    ap.add_argument("--plan-out", default="")
    args = ap.parse_args()

    plan = [cmd.to_dict() for cmd in build_plan(args.model, args.hf_id, args.layer, args.out)]
    text = json.dumps({"model": args.model, "hf_id": args.hf_id, "commands": plan}, indent=2)
    if args.plan_out:
        path = Path(args.plan_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
