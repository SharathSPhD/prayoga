"""Tier-1 live sweep (WP1.1/1.3) — runs the attack battery on Claude via claude -p.

COST: live calls. With CliLMClient lean mode ~$0.04/call; bound with --limit.
Reports per-family ASR + bootstrap CI and writes JSON.

  docker-free; runs on host:
  python -m prayoga.benchmarks.run_tier1 --limit 20 --out results/tier1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from prayoga.benchmarks.tier1_behavioral import run_battery
from prayoga.lm.cli_client import CliLMClient


def _load(path: str, limit: int | None) -> list[str]:
    reqs = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    return reqs[:limit] if limit else reqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts/harmful.txt")
    ap.add_argument("--limit", type=int, default=None, help="cap #requests (cost)")
    ap.add_argument("--model", default=None, help="pin a Claude model id")
    ap.add_argument("--out", default="results/tier1")
    args = ap.parse_args()

    requests = _load(args.prompts, args.limit)
    client = CliLMClient(model=args.model)
    print(f"[tier1] {len(requests)} requests x battery via claude -p (lean)")
    results = run_battery(client, requests)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    (out / "tier1_asr.json").write_text(json.dumps(payload, indent=2))
    print("\n=== Tier-1 ASR by family ===")
    for r in sorted(results, key=lambda x: x.asr):
        print(f"  {r.family:20s} ASR={r.asr:.2f} CI95=[{r.asr_ci[0]:.2f},{r.asr_ci[1]:.2f}] n={r.n}")
    print(f"\nwrote {out}/tier1_asr.json")


if __name__ == "__main__":
    main()
