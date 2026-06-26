"""Convergent hardening: layer-sweep of refusal-subspace effective dimension.

The review's standing Achilles heel: F8/F9's dimensionality and scale claims rest on
a SINGLE extraction layer. Here we compute the effective dimension at EVERY layer for
a model, so we can say whether dimensionality is layer-invariant (the single-layer
number was representative) or layer-specific (the claim must be qualified).

  docker exec prayoga-gpu python -m prayoga.axis_a.run_dimsweep --model gemma-2-2b-it

Claim-tier: MECHANISM (rigor check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.dimensionality import separability_dimensionality
from prayoga.lm.hf_model import HFModel


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    model = HFModel(args.hf_id)
    Hh = model.capture_all_layers_last_token(harmful)  # [L, n, d]
    Hs = model.capture_all_layers_last_token(harmless)

    per_layer = []
    for L in range(model.n_layers):
        r = separability_dimensionality(Hh[L], Hs[L])
        per_layer.append({"layer": L, "effective_dim": r["effective_dim"],
                          "full_acc": round(r["full_acc"], 3),
                          "acc_after_1": round(r["acc_after_removing_1"], 3) if r["acc_after_removing_1"] else None})
        print(f"[dimsweep] L={L:2d} eff_dim={r['effective_dim']} full={r['full_acc']:.2f} after1={r['acc_after_removing_1']}")

    dims = [x["effective_dim"] for x in per_layer]
    summary = {
        "model": args.model, "n_layers": model.n_layers,
        "eff_dim_min": int(np.min(dims)), "eff_dim_max": int(np.max(dims)),
        "eff_dim_median": float(np.median(dims)), "eff_dim_mode": int(np.bincount(dims).argmax()),
        "layer_invariant": bool(np.std(dims) < 1.0),
        "per_layer": per_layer,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"dimsweep_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== DIMENSION LAYER-SWEEP ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))


if __name__ == "__main__":
    main()
