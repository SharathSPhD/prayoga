"""WP2.C2 experiment: the turīya prompt-invariant attractor test.

For several diverse seeds, iterate self-paraphrase and ask:
  * CONVERGENCE — does each trajectory stabilize (tail step-similarity high)?
  * INVARIANCE  — do DIFFERENT seeds converge to the SAME attractor, i.e. are the
                  final states more similar to each other than unrelated texts?

Because LLM embeddings are anisotropic (random texts already cosine-similar), the
invariance gate compares final-state similarity against a BASELINE = the mean
pairwise similarity of the initial (unrelated) seeds. The strong turīya claim is
supported ONLY if cross-seed invariance exceeds that baseline by a margin; else it
is falsified (no shared prompt-invariant attractor). Never a vimarśa claim.

  docker exec prayoga-gpu python -m prayoga.axis_c.run_attractor --model gemma-2-2b-it

Claim-tier: METAPHOR-with-falsifiable-core.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_c.attractor import _cos, analyze_attractors, paraphrase_trajectory
from prayoga.lm.hf_model import HFModel

SEEDS = [
    "The old lighthouse kept watch over the storm-battered harbor every night.",
    "Quantum computers exploit superposition to evaluate many states at once.",
    "She planted tomatoes and basil along the sunny edge of the garden.",
    "The central bank raised interest rates to curb rising inflation.",
    "A lone wolf howled at the pale moon above the silent pines.",
    "Democracy depends on the free and informed participation of its citizens.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--tail", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--out", default="results/axis_c")
    args = ap.parse_args()

    model = HFModel(args.hf_id)
    init_embs = [model.embed_text(s) for s in SEEDS]
    base_pairs = [
        _cos(init_embs[i], init_embs[j])
        for i in range(len(SEEDS))
        for j in range(i + 1, len(SEEDS))
    ]
    baseline = float(np.mean(base_pairs))  # anisotropy / chance similarity

    finals, tail_sims, trajectories = [], [], []
    for s in SEEDS:
        texts, sims = paraphrase_trajectory(model, s, args.n_iter)
        finals.append(model.embed_text(texts[-1]) if texts[-1] else init_embs[0])
        tail_sims.extend(sims[-args.tail :])
        trajectories.append(texts)
        print(f"[seed] '{s[:40]}...' tail_sim={np.mean(sims[-args.tail:]):.3f}")

    res = analyze_attractors(finals, tail_sims)
    invariant = bool(res["cross_seed_invariance"] > baseline + args.margin)
    converged = bool(res["convergence"] > 0.9)
    turiya_supported = bool(invariant and converged)

    summary = {
        "model": args.model,
        "baseline_unrelated_similarity": baseline,
        "convergence_tail_sim": res["convergence"],
        "cross_seed_invariance": res["cross_seed_invariance"],
        "invariance_margin_over_baseline": res["cross_seed_invariance"] - baseline,
        "trajectories_converge": converged,
        "shared_invariant_attractor": invariant,
        "turiya_strong_claim_supported": turiya_supported,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"attractor_{args.model}.json").write_text(json.dumps(summary, indent=2))
    (out / f"attractor_{args.model}_trajectories.json").write_text(json.dumps(trajectories, indent=2))
    print("\n=== TURĪYA ATTRACTOR TEST ===")
    print(json.dumps(summary, indent=2))
    verdict = "SUPPORTED" if turiya_supported else "FALSIFIED (no shared prompt-invariant attractor)"
    print(f"\nturīya strong claim: {verdict}")


if __name__ == "__main__":
    main()
