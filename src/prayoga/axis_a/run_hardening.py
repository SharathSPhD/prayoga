"""Convergent hardening: the two most decisive honesty checks from the review.

(A) F13 circularity — is the "unsupervised" SAE feature #566 just F1's
    difference-in-means direction relabeled? Report cos(d_ref, W_dec[top]), the
    gap distribution (is #566 special or one of many), how many of the top-10
    gap features individually ablate to high ASR, and a disjoint-split check.
(B) F11 extraction-artifact — extract d_ref on ONE harmful domain (weapons) and
    measure the order-parameter F-ratio on paraphrase orbits of a DIFFERENT
    domain (cyber). If the invariance survives, it is domain-general, not an
    artifact of the extraction set.

  docker exec prayoga-gpu python -m prayoga.axis_a.run_hardening --model gemma-2-2b-it --layer 7

Claim-tier: MECHANISM (rigor check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.direction_extraction import directions_all_layers
from prayoga.axis_a.intervention_engine import InterventionEngine
from prayoga.axis_a.refusal import asr
from prayoga.axis_a.run_sae import _corpus_texts
from prayoga.axis_a.sae import feature_activations, train_sae
from prayoga.axis_b.symmetry import PARAPHRASE_STYLES, f_ratio, order_parameter
from prayoga.lm.hf_model import HFModel
from prayoga.satkarma.run_satkarma_v2 import CYBER_KW, WEAPON_KW, _cat


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def _orbit(model, seed, layer):
    paras = [model.generate([s.format(x=seed)], 40)[0].strip() or seed for s in PARAPHRASE_STYLES]
    return model.capture_all_layers_last_token(paras)[layer]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    model = HFModel(args.hf_id)
    eng = InterventionEngine(model)
    L = args.layer

    # ---- (A) F13 circularity ------------------------------------------------ #
    d_ref = directions_all_layers(model, harmful[:24], harmless[:24])[L]
    d_ref = d_ref / np.linalg.norm(d_ref)
    sae, metrics = train_sae(model.capture_positions(_corpus_texts(500), L), 4096, 32, steps=args.steps)
    dec = sae._unit_dec().detach().cpu().numpy()
    zh = feature_activations(sae, model.capture_all_layers_last_token(harmful[:24])[L]).mean(0)
    zs = feature_activations(sae, model.capture_all_layers_last_token(harmless[:24])[L]).mean(0)
    gap = zh - zs
    order = np.argsort(gap)[::-1]
    top = int(order[0])
    cos_top = float(abs(d_ref @ dec[top]))
    cos_top5 = [round(float(abs(d_ref @ dec[int(i)])), 3) for i in order[:5]]
    # how many of the top-10 gap features individually fully ablate refusal?
    h_ev = harmful[24:]
    base = asr(eng.baseline_generate(h_ev, 32))
    indiv = []
    for i in order[:10]:
        indiv.append(round(asr(eng.ablate_generate(h_ev, dec[int(i)], 32)), 3))
    n_strong = sum(1 for a in indiv if a > 0.8)
    gap_concentration = float(gap[order[0]] / (gap[order[:10]].sum() + 1e-9))

    # ---- (B) F11 extraction-artifact (held-out cross-domain) ---------------- #
    weap = _cat(harmful, WEAPON_KW)
    cyber = _cat(harmful, CYBER_KW)
    d_weap = directions_all_layers(model, weap, harmless[:len(weap)])[L]
    d_weap = d_weap / np.linalg.norm(d_weap)
    rng = np.random.RandomState(0)
    d_rand = rng.randn(model.d_model); d_rand /= np.linalg.norm(d_rand)
    # orbits of CYBER harmful (label 1) and harmless (label 0)
    seeds = [(s, 1) for s in cyber[:6]] + [(s, 0) for s in harmless[24:30]]
    orbit_acts, labels = [], []
    for s, lab in seeds:
        orbit_acts.append(_orbit(model, s, L)); labels.append(lab)
    ms_ref = [order_parameter(a, d_weap) for a in orbit_acts]
    ms_rand = [order_parameter(a, d_rand) for a in orbit_acts]
    fr_cross = f_ratio(ms_ref, labels)
    fr_rand = f_ratio(ms_rand, labels)

    summary = {
        "model": args.model, "layer": L, "sae_FVU": round(metrics["fvu"], 4),
        "F13_circularity": {
            "cos_dref_topfeature": round(cos_top, 3),
            "cos_dref_top5_features": cos_top5,
            "top10_individual_ablation_ASR": indiv,
            "n_top10_features_strong(>0.8)": n_strong,
            "gap_concentration_top1_of_top10": round(gap_concentration, 3),
            "verdict": ("REDISCOVERY of F1 direction" if cos_top > 0.8
                        else "DISTINCT from F1 direction" if cos_top < 0.5
                        else "PARTIAL overlap with F1"),
        },
        "F11_holdout_cross_domain": {
            "extracted_on": "weapons", "tested_on_orbits": "cyber+harmless",
            "F_ratio_weapons_dir_on_cyber": round(fr_cross, 3),
            "F_ratio_random": round(fr_rand, 3),
            "invariance_is_domain_general": bool(fr_cross > 3 * max(fr_rand, 1e-6) and fr_cross > 2),
        },
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"hardening_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== CONVERGENT HARDENING ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
