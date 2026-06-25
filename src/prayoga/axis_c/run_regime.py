"""WP2.C1 experiment: avasthātraya regime-probe with transfer + null gates.

Captures residual activations for jāgrat / svapna / suṣupti prompt sets, then
probes every layer. The HEADLINE gate uses a PRE-SPECIFIED layer (network
middle) to avoid the layer-selection bias flagged in Axis-A review; the full
per-layer profile is reported for transparency.

  docker exec prayoga-gpu python -m prayoga.axis_c.run_regime --model gemma-2-2b-it

Claim-tier: METAPHOR-with-falsifiable-core.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prayoga.axis_c.regime_probes import evaluate_regime_probe
from prayoga.lm.hf_model import HFModel


def _load(p: str) -> list[str]:
    return [l.rstrip("\n") for l in Path(p).read_text().splitlines() if l.strip("\n")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--out", default="results/axis_c")
    args = ap.parse_args()

    regimes = {
        "jagrat": _load("data/prompts/regimes/jagrat.txt"),
        "svapna": _load("data/prompts/regimes/svapna.txt"),
        "susupti": _load("data/prompts/regimes/susupti.txt"),
    }
    print({k: len(v) for k, v in regimes.items()})
    model = HFModel(args.hf_id)
    # [regime] -> [L, n, d]
    acts = {r: model.capture_all_layers_last_token(p) for r, p in regimes.items()}

    # cheap profile (no null) at every layer
    per_layer = []
    for L in range(model.n_layers):
        res = evaluate_regime_probe({r: acts[r][L] for r in regimes}, n_shuffle=0)
        per_layer.append({"layer": L, "transfer_acc": res["transfer_acc"]})
        print(f"[probe] L={L:2d} transfer_acc={res['transfer_acc']:.2f}")

    mid = model.n_layers // 2  # PRE-SPECIFIED headline layer
    best_layer = max(per_layer, key=lambda x: x["transfer_acc"])["layer"]
    # full null only where it matters (headline + best-for-info)
    full = {
        L: evaluate_regime_probe({r: acts[r][L] for r in regimes}, n_shuffle=300)
        for L in {mid, best_layer}
    }
    headline = {"layer": mid, "transfer_acc": full[mid]["transfer_acc"],
                "null_p": full[mid]["null_p"], "passed": full[mid]["passed"]}
    best = {"layer": best_layer, "transfer_acc": full[best_layer]["transfer_acc"],
            "null_p": full[best_layer]["null_p"], "passed": full[best_layer]["passed"]}
    summary = {
        "model": args.model,
        "regimes": list(regimes),
        "chance": round(1.0 / len(regimes), 3),
        "headline_layer": mid,
        "headline_transfer_acc": headline["transfer_acc"],
        "headline_null_p": headline["null_p"],
        "headline_PASS": headline["passed"],
        "best_layer_for_info": best["layer"],
        "best_transfer_acc": best["transfer_acc"],
        "per_layer": per_layer,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"regime_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== REGIME PROBE ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))
    print(f"\nHEADLINE (layer {mid}) {'PASS' if headline['passed'] else 'FAIL/DEMOTE'}")


if __name__ == "__main__":
    main()
