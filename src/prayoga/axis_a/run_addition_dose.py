"""Powered addition dose-response — nails down F23 (the addition asymmetry is dosing).

Sweeps a fine coefficient grid for single-direction addition of the diff-in-means
refusal direction at the extraction layer, on n harmless prompts, and records the
over-refusal rate as a function of coefficient. The prediction (F23): Qwen shows an
inverted-U — over-refusal rises to ~1 at a low coefficient then falls as larger
coefficients push activations off-distribution into incoherent text; Gemma peaks at a
larger coefficient. The peak coefficient, not sufficiency, is what differs across
families. Raw generations are dumped git-ignored for host-side judge verification of
the peak cell (`judge` confirms peak over-refusal is genuine, not garbage).

  docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
    python -m prayoga.axis_a.run_addition_dose --model qwen2.5-3b-it --hf-id Qwen/Qwen2.5-3B-Instruct --layer 19

Aggregate-only public output. Claim-tier: MECHANISM.
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
    ap.add_argument("--coeffs", default="0,2,4,8,12,16,24,32,48,64,96,128")
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    L, nt = args.layer, args.n_train
    model = HFModel(args.hf_id)

    Xh = model.capture_all_layers_last_token(harmful[:nt])[L]
    Xs = model.capture_all_layers_last_token(harmless[:nt])[L]
    d_ref = diff_in_means(Xh, Xs)
    align = probe_alignment(Xh, Xs)

    ev = harmless[nt:]  # 24 harmless eval prompts
    coeffs = [float(c) for c in args.coeffs.split(",")]
    curve = []
    raw_cells = {}
    for c in coeffs:
        if c == 0:
            outs = model.generate(ev, 40)
        else:
            with model.addition_hooks(d_ref, coeff=c, layer=L):
                outs = model.generate(ev, 40)
        curve.append({"coeff": c, "over_refusal": round(refusal_rate(outs), 3)})
        raw_cells[str(c)] = outs

    peak = max(curve, key=lambda r: r["over_refusal"])
    summary = {
        "model": args.model, "layer": L, "n_eval": len(ev),
        **align,
        "peak_coeff": peak["coeff"], "peak_over_refusal": peak["over_refusal"],
        "over_refusal_at_64": next((r["over_refusal"] for r in curve if r["coeff"] == 64), None),
        "curve": curve,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"addition_dose_{args.model}.json").write_text(json.dumps(summary, indent=2))
    # dual-use raw dump for host-side judge verification of the peak cell
    (out / f"_raw_addition_dose_{args.model}.json").write_text(
        json.dumps({"requests": ev, "peak_coeff": peak["coeff"], "cells": raw_cells})
    )
    print("\n=== ADDITION DOSE-RESPONSE (F23 powered) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
