"""WP-B1 experiment: the ṣaṭkarma intervention taxonomy on a real model.

Extracts the refusal direction, a category-specific direction, and the residual
principal components, then runs all six acts (each with a control) and writes a
taxonomy table. Every number is a real measurement.

  docker exec prayoga-gpu python -m prayoga.satkarma.run_satkarma --model gemma-2-2b-it --layer 7

Claim-tier: METAPHOR-with-falsifiable-core (the taxonomy is the testable object).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.direction_extraction import directions_all_layers
from prayoga.axis_a.intervention_engine import InterventionEngine
from prayoga.lm.hf_model import HFModel
from prayoga.satkarma import operators as ops


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def _pcs(model: HFModel, prompts: list[str], layer: int, k: int) -> np.ndarray:
    """Top-k residual principal components at a layer (centered SVD)."""
    A = model.capture_all_layers_last_token(prompts)[layer].astype(np.float64)
    A = A - A.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:k]  # [k, d]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--mnt", type=int, default=32)
    ap.add_argument("--add-mult", type=float, default=8.0)
    ap.add_argument("--marana-k", type=int, default=10)
    ap.add_argument("--out", default="results/satkarma")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    questions = _load("data/prompts/regimes/jagrat.txt")
    L, nt = args.layer, args.n_train
    h_tr, h_ev = harmful[:nt], harmful[nt:]
    s_tr, s_ev = harmless[:nt], harmless[nt:]

    model = HFModel(args.hf_id)
    eng = InterventionEngine(model)
    print(f"[satkarma] {args.model} L={L} d={model.d_model}")

    # directions
    d_ref = directions_all_layers(model, h_tr, s_tr)[L]
    # category split for uccāṭana: first half of harmful = "target category"
    cat_a, cat_b = harmful[: len(harmful) // 2], harmful[len(harmful) // 2 :]
    d_cat = directions_all_layers(model, cat_a[:nt], s_tr)[L]
    # addition scale
    h_acts = model.capture_all_layers_last_token(h_tr)[L]
    s_acts = model.capture_all_layers_last_token(s_tr)[L]
    sep = float(np.linalg.norm(h_acts.mean(0) - s_acts.mean(0)))
    coeff = args.add_mult * sep
    # principal components for stambhana / māraṇa
    pcs = _pcs(model, harmful + harmless, L, args.marana_k)

    results = [
        ops.vasikarana(eng, d_ref, h_ev, args.mnt),
        ops.santi(eng, d_ref, s_ev, L, coeff, args.mnt),
        ops.stambhana(eng, pcs[0], s_ev[:12], args.mnt),
        ops.vidvesana(eng, d_ref, questions[:12], L, coeff, args.mnt),
        ops.uccatana(eng, d_cat, cat_a[nt // 2 :], cat_b[: len(cat_a) - nt // 2], args.mnt),
        ops.marana(eng, pcs, s_ev[:12], args.mnt),
    ]
    table = [r.to_dict() for r in results]
    n_sep = sum(r.separated for r in results)
    summary = {
        "model": args.model, "layer": L, "add_coeff": round(coeff, 1),
        "n_acts_separated": n_sep, "n_acts": len(results),
        "taxonomy_supported": n_sep >= 4, "table": table,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"satkarma_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ṢAṬKARMA TAXONOMY ===")
    for r in results:
        flag = "✓" if r.separated else "·"
        print(f" {flag} {r.sanskrit:12s} {r.metric:38s} effect={r.effect:.3f} control={r.control:.3f}")
    print(f"\n{n_sep}/6 acts control-separated → taxonomy {'SUPPORTED' if n_sep>=4 else 'WEAK'}")


if __name__ == "__main__":
    main()
