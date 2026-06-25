"""Run a redacted multi-turn refusal trajectory probe.

This runner is local/DGX-oriented and intentionally prints only aggregate turn
metrics. Raw harmful prompts, injection payloads, completions, and vectors should
remain in private result paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_b.trajectory import analyze_refusal_trajectory
from prayoga.benchmarks.attacks import Crescendo
from prayoga.benchmarks.tier1_behavioral import is_hallucination_proxy
from prayoga.lm.hf_model import HFModel
from prayoga.shared.refusal import is_refusal


def _load_direction(path: str) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("direction", data.get("d_ref"))
    if data is None:
        raise ValueError("direction JSON must be an array or contain `direction` / `d_ref`")
    return np.asarray(data, dtype=float)


def _load_requests(path: str, limit: int) -> list[str]:
    requests = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    return requests[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--direction", required=True, help="private JSON file containing the direction")
    ap.add_argument("--requests", default="data/prompts/harmful.txt")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="results/axis_b/trajectory.json")
    args = ap.parse_args()

    direction = _load_direction(args.direction)
    requests = _load_requests(args.requests, args.limit)
    model = HFModel(args.hf_id)
    attack = Crescendo()
    summaries = []

    for idx, request in enumerate(requests):
        transcript = ""
        labels, acts, refused, hallucination = [], [], [], []
        for turn_idx, turn in enumerate(attack.render(request)):
            prompt = (transcript + f"\nUser: {turn}\nAssistant:").strip()
            all_layers = model.capture_all_layers_last_token([prompt])
            acts.append(all_layers[args.layer, 0])
            reply = model.generate([prompt], max_new_tokens=args.max_new_tokens)[0]
            labels.append(f"turn_{turn_idx}")
            refused.append(is_refusal(reply))
            hallucination.append(is_hallucination_proxy(reply))
            transcript = f"{prompt} {reply}"

        result = analyze_refusal_trajectory(
            np.asarray(acts),
            direction,
            attack="crescendo",
            labels=labels,
            refused=refused,
            hallucination_proxy=hallucination,
        )
        row = result.to_dict()
        row["request_index"] = idx
        summaries.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "hf_id": args.hf_id,
        "layer": args.layer,
        "attack": "crescendo",
        "redacted": True,
        "n_requests": len(summaries),
        "trajectories": summaries,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    public = {
        "model": args.model,
        "layer": args.layer,
        "attack": "crescendo",
        "n_requests": len(summaries),
        "mean_collapse": float(np.mean([s["collapse"] for s in summaries])) if summaries else 0.0,
        "n_recovered": int(sum(1 for s in summaries if s["recovered"])),
    }
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
