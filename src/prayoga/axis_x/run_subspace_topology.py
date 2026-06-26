"""WP-X1 sub-study: refusal-subspace necessary-vs-sufficient geometry (GPU).

For each model, at its extraction layer, measures whether the single diff-in-means
ablation axis spans the refusal subspace (residual harmful/harmless separability
after removing it). Collapse-to-chance ⇒ addition along that axis is sufficient
(Gemma-like); survival ⇒ orthogonal sufficiency structure remains (Qwen-like).
This is the geometric correlate of the F6 addition asymmetry, and the returned
scalars are comparable across different-residual-width models (where principal
angles are undefined — see subspace_topology.py).

  docker exec prayoga-gpu python -m prayoga.axis_x.run_subspace_topology

Aggregate-only output (CV accuracies, effective dims, angles). Bases are never written.
Claim-tier: MECHANISM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prayoga.axis_a.direction_extraction import directions_all_layers
from prayoga.axis_x.subspace_topology import necessary_vs_sufficient
from prayoga.lm.hf_model import HFModel

# (short name, HF id, extraction layer) — layers from results/axis_a/direction_*.json
MODELS = [
    ("gemma-2-2b-it", "google/gemma-2-2b-it", 7),
    ("qwen2.5-3b-it", "Qwen/Qwen2.5-3B-Instruct", 19),
    ("gemma-2-9b-it", "google/gemma-2-9b-it", 10),
]


def _load(p: str) -> list[str]:
    return [ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="results/axis_x")
    ap.add_argument("--only", default="", help="comma-separated short names to restrict to")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    nt = args.n_train
    only = {s for s in args.only.split(",") if s}

    per_model = {}
    for short, hf_id, layer in MODELS:
        if only and short not in only:
            continue
        print(f"\n--- {short} (L{layer}) ---", flush=True)
        model = HFModel(hf_id)
        d_ref = directions_all_layers(model, harmful[:nt], harmless[:nt])[layer]
        Xh = model.capture_all_layers_last_token(harmful[:nt])[layer]
        Xs = model.capture_all_layers_last_token(harmless[:nt])[layer]
        geom = necessary_vs_sufficient(Xh, Xs, d_ref, k=args.k)
        geom["layer"] = layer
        geom["d_model"] = int(model.d_model)
        per_model[short] = geom
        print(json.dumps(geom, indent=2), flush=True)
        del model  # free GPU before loading the next

    summary = {
        "per_model": per_model,
        "note": (
            "Cross-family principal angles are undefined (different residual widths); "
            "comparison is by residual-separability and effective-dim scalars."
        ),
        "interpretation": (
            "Models whose ablation axis spans refusal (residual CV acc → chance) admit "
            "sufficient single-direction addition (F6 Gemma side); models retaining "
            "orthogonal separability after removal do not (F6 Qwen side)."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "subspace_topology.json").write_text(json.dumps(summary, indent=2))
    print("\n=== REFUSAL-SUBSPACE GEOMETRY (necessary vs sufficient) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
