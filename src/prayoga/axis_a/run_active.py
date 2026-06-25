"""WP2.A6 experiment: active-inference refusal-circuit discovery on SAE features.

Trains the SAE, takes the top-M candidate features by harmful-vs-harmless gap, and
compares three strategies for building a minimal ablation circuit under a fixed
intervention budget: active (EFE-style pragmatic*diversity), greedy (static gap),
random. Reports the ASR-vs-interventions curve and the oracle efficiency
(interventions to reach 80% of the max attainable ASR) for each.

  docker exec prayoga-gpu python -m prayoga.axis_a.run_active --model gemma-2-2b-it --layer 7

Claim-tier: MECHANISM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.active_discovery import discover_circuit
from prayoga.axis_a.intervention_engine import InterventionEngine
from prayoga.axis_a.refusal import asr
from prayoga.axis_a.run_sae import _corpus_texts
from prayoga.axis_a.sae import feature_activations, train_sae
from prayoga.lm.hf_model import HFModel


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--n-features", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--cand", type=int, default=40)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--mnt", type=int, default=32)
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    model = HFModel(args.hf_id)
    eng = InterventionEngine(model)
    L = args.layer

    print("[active] training SAE ...")
    sae, metrics = train_sae(model.capture_positions(_corpus_texts(500), L),
                             args.n_features, args.k, steps=args.steps)
    dec = sae._unit_dec().detach().cpu().numpy()
    print(f"[active] SAE FVU={metrics['fvu']:.3f}")

    zh = feature_activations(sae, model.capture_all_layers_last_token(harmful[:24])[L]).mean(0)
    zs = feature_activations(sae, model.capture_all_layers_last_token(harmless[:24])[L]).mean(0)
    gap = zh - zs
    cand = list(np.argsort(gap)[::-1][: args.cand].astype(int))

    h_ev = harmful[24:][: args.n_eval]

    def measure_asr(dirs: np.ndarray) -> float:
        if dirs.shape[0] == 0:
            return asr(eng.baseline_generate(h_ev, args.mnt))
        return asr(model.generate_with_subspace_ablation(h_ev, dirs, 1.0, args.mnt))

    results = {}
    for strat in ("active", "greedy", "random"):
        r = discover_circuit(cand, dec, gap, measure_asr, strategy=strat, budget=args.budget)
        results[strat] = r
        print(f"[{strat:7s}] curve={r['asr_curve']}")
    # GLOBAL target: 80% of the best ASR any strategy achieved
    gmax = max(max(results[s]["asr_curve"]) for s in results)
    target = 0.8 * gmax
    for s in results:
        curve = results[s]["asr_curve"]
        results[s]["interventions_to_target"] = next(
            (k for k, a in enumerate(curve) if a >= target), args.budget + 1)
        results[s]["final_asr"] = curve[-1]
        print(f"  {s:7s}: to-{target:.2f} = {results[s]['interventions_to_target']}, final ASR={curve[-1]}")

    summary = {
        "model": args.model, "layer": L, "sae_FVU": round(metrics["fvu"], 4),
        "n_candidates": args.cand, "budget": args.budget,
        "active_to_target": results["active"]["interventions_to_target"],
        "greedy_to_target": results["greedy"]["interventions_to_target"],
        "random_to_target": results["random"]["interventions_to_target"],
        "final_asr": {s: results[s]["final_asr"] for s in results},
        "active_beats_random": results["active"]["interventions_to_target"]
        < results["random"]["interventions_to_target"],
        "strategies": results,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"active_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ACTIVE CIRCUIT DISCOVERY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "strategies"}, indent=2))


if __name__ == "__main__":
    main()
