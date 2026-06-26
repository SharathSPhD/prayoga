"""WP-X1 experiment: cross-axis same-object triangulation (GPU).

On ONE model and ONE set of harmful prompts, applies a single refusal-suppression
injection and measures all three axes per prompt — Axis A (refusal order
parameter), Axis B (monitoring-precision margin), Axis C (dormant-baseline
distance) — together with the behavioural refuse→comply flip. Asks whether the
internal collapse predicts behaviour beyond a random-direction control and a
label-shuffle null (the X-1 keystone gate).

  docker exec prayoga-gpu python -m prayoga.axis_x.run_triangulation --model gemma-2-2b-it --layer 7
  docker exec prayoga-gpu python -m prayoga.axis_x.run_triangulation --hf-id Qwen/Qwen2.5-3B-Instruct --model qwen2.5-3b-it --layer 19

Aggregate-only output (scalars, CIs, gate booleans). No raw vectors are written.
Claim-tier: MECHANISM↔ANALOGY boundary; Axis C is descriptive and excluded from the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.direction_extraction import directions_all_layers
from prayoga.axis_b.precision import train_refusal_probe
from prayoga.axis_c.susupti import SUSUPTI_BASELINES
from prayoga.axis_x.triangulation import couple, deltas_from_activations
from prayoga.benchmarks.attacks import (
    IndirectInjection,
    ManyShot,
    PersonaDAN,
    RefusalSuppression,
)
from prayoga.lm.hf_model import HFModel
from prayoga.shared.refusal import refusal_flags

# Single-turn injection families pooled to give graded behavioural variance.
INJECTION_FAMILIES = [RefusalSuppression(), PersonaDAN(), ManyShot(), IndirectInjection()]


def _load(p: str) -> list[str]:
    return [ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--out", default="results/axis_x")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    L, nt = args.layer, args.n_train
    model = HFModel(args.hf_id)

    # refusal direction (diff-in-means) — recomputed, never persisted (dual-use)
    d_ref = directions_all_layers(model, harmful[:nt], harmless[:nt])[L]

    # monitoring-precision probe at the extraction layer
    Xh = model.capture_all_layers_last_token(harmful[:nt])[L]
    Xs = model.capture_all_layers_last_token(harmless[:nt])[L]
    probe = train_refusal_probe(Xh, Xs)

    # evaluation prompts: DISJOINT from probe/direction training split (no leakage)
    h_ev = harmful[nt:]
    acts_base = model.capture_all_layers_last_token(h_ev)[L]
    plain_outputs = model.generate(h_ev, 60)
    refused_plain = refusal_flags(plain_outputs)

    # Axis-C dormant baseline (neutral_dormant suṣupti prompt)
    dormant_prompt = next(b.prompt for b in SUSUPTI_BASELINES if b.name == "neutral_dormant")
    dormant_mean = model.capture_all_layers_last_token([dormant_prompt])[L][0]

    rng = np.random.RandomState(0)
    rand_dirs = rng.normal(0, 1, (args.n_random, model.d_model))
    rand_dirs = rand_dirs / np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    # Pool (prompt × injection-family) pairs: any injected-context suppression should
    # couple internal collapse to behaviour. Baseline acts/behaviour are shared per prompt.
    dA, dB, dC, base_strength, dA_rand, flip, groups = [], [], [], [], [], [], []
    per_family = {}
    raw_families = []
    for gid, fam in enumerate(INJECTION_FAMILIES):
        inj = [fam.render(p)[0] for p in h_ev]
        acts_inj = model.capture_all_layers_last_token(inj)[L]
        inj_outputs = model.generate(inj, 60)
        refused_inj = refusal_flags(inj_outputs)
        fam_flip = refused_plain.astype(bool) & ~refused_inj.astype(bool)
        d = deltas_from_activations(
            acts_base, acts_inj, d_ref,
            probe=probe, dormant_mean=dormant_mean, random_directions=rand_dirs,
        )
        dA.append(d["delta_A"])
        dB.append(d["delta_B"])
        dC.append(d["delta_C"])
        base_strength.append(d["baseline_strength"])
        dA_rand.append(d["delta_A_random"])
        flip.append(fam_flip)
        groups.append(np.full(len(h_ev), gid))
        per_family[fam.name] = {
            "flip_rate": round(float(fam_flip.mean()), 3),
            "refusal_rate_injected": round(float(refused_inj.mean()), 3),
            "delta_A_mean": round(float(d["delta_A"].mean()), 4),
            "delta_B_mean": round(float(d["delta_B"].mean()), 4),
        }
        raw_families.append({
            "name": fam.name,
            "gid": gid,
            "injected_outputs": inj_outputs,
            "delta_A": d["delta_A"].tolist(),
            "delta_B": d["delta_B"].tolist(),
            "delta_C": d["delta_C"].tolist(),
            "baseline_strength": d["baseline_strength"].tolist(),
            "delta_A_random": d["delta_A_random"].tolist(),
        })

    res = couple(
        np.concatenate(dA), np.concatenate(dB), np.concatenate(dC), np.concatenate(flip),
        baseline_strength=np.concatenate(base_strength),
        delta_A_random=np.concatenate(dA_rand),
        groups=np.concatenate(groups),
        model=args.model, layer=L,
    )
    summary = res.to_dict()
    summary["refusal_rate_plain"] = round(float(refused_plain.mean()), 3)
    summary["injection_families"] = [f.name for f in INJECTION_FAMILIES]
    summary["per_family"] = per_family

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"triangulation_{args.model}.json").write_text(json.dumps(summary, indent=2))

    # Dual-use raw dump (git-ignored under results/): deltas + generations + requests,
    # consumed by the host-side content judge (score_triangulation.py). Never published.
    raw = {
        "model": args.model, "layer": L, "requests": h_ev,
        "plain_outputs": plain_outputs, "families": raw_families,
    }
    (out / f"_raw_triangulation_{args.model}.json").write_text(json.dumps(raw))

    print("\n=== X-1 CROSS-AXIS TRIANGULATION (substring DV) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
