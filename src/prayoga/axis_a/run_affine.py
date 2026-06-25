"""S3 GPU runner: the affine source of the F6 addition asymmetry.

For each model: (1) measure cos(d̂, w) between the diff-in-means direction and the
refusal-probe normal — the geometric predictor of whether single-direction addition can
induce refusal; (2) sweep (layer, coefficient) addition of d̂ onto harmless prompts and
measure the over-refusal rate, to confirm behaviourally whether ANY configuration makes
Qwen over-refuse (vs Gemma).

  docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
    python -m prayoga.axis_a.run_affine --model qwen2.5-3b-it --hf-id Qwen/Qwen2.5-3B-Instruct --layer 19

Aggregate-only output. Claim-tier: MECHANISM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prayoga.axis_a.affine import diff_in_means, probe_alignment
from prayoga.lm.hf_model import HFModel
from prayoga.shared.refusal import refusal_rate


def _load(p: str) -> list[str]:
    return [ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--coeffs", default="4,8,16,32,64")
    ap.add_argument("--layer-span", type=int, default=3, help="+/- layers around --layer to sweep")
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    L, nt = args.layer, args.n_train
    model = HFModel(args.hf_id)

    Xh = model.capture_all_layers_last_token(harmful[:nt])[L]
    Xs = model.capture_all_layers_last_token(harmless[:nt])[L]
    align = probe_alignment(Xh, Xs)
    d_ref = diff_in_means(Xh, Xs)

    # behavioural confirmation: does adding d̂ induce over-refusal on harmless prompts?
    harmless_ev = harmless[nt:]
    base_over_refuse = refusal_rate(model.generate(harmless_ev, 40))
    coeffs = [float(c) for c in args.coeffs.split(",")]
    layers = [li for li in range(L - args.layer_span, L + args.layer_span + 1) if 0 <= li < model.n_layers]

    sweep = []
    best = {"over_refusal_delta": -1.0}
    for li in layers:
        for c in coeffs:
            with model.addition_hooks(d_ref, coeff=c, layer=li):
                over = refusal_rate(model.generate(harmless_ev, 40))
            delta = over - base_over_refuse
            rec = {"layer": li, "coeff": c, "over_refusal": round(over, 3),
                   "over_refusal_delta": round(delta, 3)}
            sweep.append(rec)
            if delta > best["over_refusal_delta"]:
                best = rec

    summary = {
        "model": args.model, "layer": L,
        "base_harmless_refusal_rate": round(base_over_refuse, 3),
        **align,
        "best_addition": best,
        "addition_can_induce_refusal": bool(best["over_refusal_delta"] > 0.3),
        "verdict": (
            "addition_sufficient" if best["over_refusal_delta"] > 0.3
            else "addition_insufficient_marshall_like"
        ),
        "sweep": sweep,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"affine_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== AFFINE GEOMETRY (source-of-truth for F6/F21 addition asymmetry) ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "sweep"}, indent=2))


if __name__ == "__main__":
    main()
