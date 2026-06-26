"""WP2.C1′ (F5): content-controlled jāgrat-vs-svapna state probe.

Fixes the surface confound that demoted the naive regime probe (F3). The factual
QUESTION is held fixed; only the internal regime varies:
  * jāgrat  = the model answers truthfully
  * svapna  = the model confabulates a confident fabricated answer to the SAME q
Activations are captured at the last token of the model's OWN generated answer.

Three gates (objectives §0 + the F3 lesson):
  1. transfer: held-out questions      (invariance across surface prompts)
  2. null:     label-shuffled          (signal is real)
  3. DEPTH:    mid-layer acc > layer-0 acc + margin   (depth-emergent, NOT
               reducible to surface token features) — the gate F3 failed.

Claim-tier: METAPHOR-with-falsifiable-core.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_c.regime_probes import evaluate_regime_probe
from prayoga.lm.hf_model import HFModel

CONFAB_FRAME = (
    "You are writing fiction. Give a confident, specific, detailed answer to the "
    "question below. It does NOT need to be true — invent plausible-sounding "
    "details. Never say you are unsure. Question: {q}"
)
TRUTH_FRAME = "Answer accurately and concisely. Question: {q}"


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.05, help="depth-gain margin over layer 0")
    ap.add_argument("--out", default="results/axis_c")
    args = ap.parse_args()

    qs = _load("data/prompts/regimes/jagrat.txt")
    model = HFModel(args.hf_id)
    print(f"[state] {len(qs)} questions x (truthful, confab) on {args.model}")

    t_acts, c_acts = [], []
    for q in qs:
        ids_t = model.generate_ids(TRUTH_FRAME.format(q=q), args.max_new_tokens)
        t_acts.append(model.capture_ids_all_layers_last_token(ids_t))  # [L,d]
        ids_c = model.generate_ids(CONFAB_FRAME.format(q=q), args.max_new_tokens)
        c_acts.append(model.capture_ids_all_layers_last_token(ids_c))

    T = np.stack(t_acts)  # [n, L, d]
    C = np.stack(c_acts)
    n_layers = T.shape[1]

    # cheap per-layer transfer profile
    prof = []
    for L in range(n_layers):
        res = evaluate_regime_probe({"truthful": T[:, L], "confab": C[:, L]}, n_shuffle=0)
        prof.append({"layer": L, "transfer_acc": res["transfer_acc"]})
        print(f"[state] L={L:2d} acc={res['transfer_acc']:.2f}")

    layer0_acc = prof[0]["transfer_acc"]
    mid = n_layers // 2
    best = max(prof, key=lambda x: x["transfer_acc"])
    full = {
        L: evaluate_regime_probe({"truthful": T[:, L], "confab": C[:, L]}, n_shuffle=300)
        for L in {0, mid, best["layer"]}
    }
    # PRINCIPLED gate: a genuine internal state must emerge MID-network, not at
    # the output-proximal final layer (which tracks surface/vocabulary). Headline
    # uses the mid layer; best layer is reported for information only.
    mid_gain = full[mid]["transfer_acc"] - layer0_acc
    headline_pass = bool(
        full[mid]["null_p"] < 0.05 and mid_gain > args.margin and full[mid]["transfer_acc"] > 0.5
    )
    summary = {
        "model": args.model,
        "n": len(qs),
        "chance": 0.5,
        "layer0_acc": layer0_acc,
        "mid_layer": mid,
        "mid_acc": full[mid]["transfer_acc"],
        "mid_null_p": full[mid]["null_p"],
        "mid_depth_gain_over_layer0": mid_gain,
        "MID_DEPTH_GATE_PASS": headline_pass,
        "best_layer_for_info": best["layer"],
        "best_acc_for_info": best["transfer_acc"],
        "per_layer": prof,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"state_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== CONTENT-CONTROLLED STATE PROBE ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))
    print(f"\nMID-LAYER DEPTH GATE {'PASS (depth-emergent state signature)' if headline_pass else 'FAIL (no mid-network state beyond surface)'}")


if __name__ == "__main__":
    main()
