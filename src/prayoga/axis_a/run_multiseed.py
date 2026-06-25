"""Convergent hardening: multi-seed replication of the primary claim (F1).

The review's most damning statistical critique was single-seed, small-n. Here we
re-run the refusal-direction ablation/addition gates across K random splits
(seeds) of the prompt sets and report per-seed effects plus the across-seed
mean±sd and a bootstrap CI pooled over seeds — separating signal from
seed-specific artifact.

  docker exec prayoga-gpu python -m prayoga.axis_a.run_multiseed --model gemma-2-2b-it --layer 7 --seeds 3

Claim-tier: MECHANISM (rigor check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.direction_extraction import directions_all_layers
from prayoga.axis_a.intervention_engine import InterventionEngine
from prayoga.axis_a.refusal import asr, is_refusal, refusal_rate
from prayoga.lm.hf_model import HFModel


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--mnt", type=int, default=32)
    ap.add_argument("--add-mult", type=float, default=8.0)
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    model = HFModel(args.hf_id)
    eng = InterventionEngine(model)
    L, nt = args.layer, args.n_train

    abl_asr, add_over, all_abl_flags = [], [], []
    for seed in range(args.seeds):
        rng = np.random.RandomState(seed)
        hi = list(rng.permutation(harmful)); si = list(rng.permutation(harmless))
        h_tr, h_ev = hi[:nt], hi[nt:]
        s_tr, s_ev = si[:nt], si[nt:]
        d = directions_all_layers(model, h_tr, s_tr)[L]
        # ablation on harmful eval
        gen = eng.ablate_generate(h_ev, d, args.mnt)
        a = asr(gen); abl_asr.append(a)
        all_abl_flags.extend([0 if is_refusal(g) else 1 for g in gen])
        # addition on harmless eval
        h_acts = model.capture_all_layers_last_token(h_tr)[L]
        s_acts = model.capture_all_layers_last_token(s_tr)[L]
        sep = float(np.linalg.norm(h_acts.mean(0) - s_acts.mean(0)))
        over = refusal_rate(eng.add_generate(s_ev, d, args.add_mult * sep, L, args.mnt))
        add_over.append(over)
        print(f"[seed {seed}] ablation ASR={a:.3f} addition over-refusal={over:.3f}")

    # pooled bootstrap CI over all eval items (across seeds)
    flags = np.array(all_abl_flags, dtype=float)
    rngb = np.random.RandomState(0)
    boot = np.array([flags[rngb.randint(0, len(flags), len(flags))].mean() for _ in range(10000)])
    summary = {
        "model": args.model, "layer": L, "seeds": args.seeds,
        "ablation_ASR_per_seed": [round(x, 3) for x in abl_asr],
        "ablation_ASR_mean": round(float(np.mean(abl_asr)), 3),
        "ablation_ASR_sd": round(float(np.std(abl_asr)), 3),
        "ablation_ASR_pooled_CI95": [round(float(np.percentile(boot, 2.5)), 3),
                                     round(float(np.percentile(boot, 97.5)), 3)],
        "addition_overrefusal_per_seed": [round(x, 3) for x in add_over],
        "addition_overrefusal_mean": round(float(np.mean(add_over)), 3),
        "stable_across_seeds": bool(np.std(abl_asr) < 0.1 and np.std(add_over) < 0.1),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"multiseed_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== MULTI-SEED REPLICATION ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
