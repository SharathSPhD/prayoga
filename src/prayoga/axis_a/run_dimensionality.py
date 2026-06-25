"""WP2.A3 runner: refusal-subspace dimensionality at the extraction layer.

  docker exec prayoga-gpu python -m prayoga.axis_a.run_dimensionality \
      --hf-id google/gemma-2-2b-it --model gemma-2-2b-it --layer 7

Fast (no generation): captures harmful/harmless activations, then iterative
logistic projection. Run per model to compare (F6 follow-up).

Claim-tier: MECHANISM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prayoga.axis_a.dimensionality import separability_dimensionality
from prayoga.lm.hf_model import HFModel


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    model = HFModel(args.hf_id)
    Xh = model.capture_all_layers_last_token(harmful)[args.layer]
    Xs = model.capture_all_layers_last_token(harmless)[args.layer]
    res = separability_dimensionality(Xh, Xs)
    res = {"model": args.model, "layer": args.layer, **res}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"dim_{args.model}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n{args.model} L{args.layer}: effective_dim={res['effective_dim']} "
          f"(full={res['full_acc']:.2f}, after-1-removed={res['acc_after_removing_1']:.2f})")


if __name__ == "__main__":
    main()
