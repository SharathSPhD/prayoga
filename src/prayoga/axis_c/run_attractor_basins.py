"""S2 source-of-truth for F7: if not ONE turīya attractor, what IS the structure?

F7 falsified the universal prompt-invariant attractor: self-paraphrase converges
per-seed (tail similarity ~0.996) but different seeds reach DIFFERENT fixed points
(cross-seed 0.952 < anisotropy baseline 0.960). So there is no single content-free
turīya. The source-of-truth question: are the multiple fixed points **content-determined
semantic basins**? We iterate self-paraphrase from several seeds grouped into two
distinct topics and test whether final-state similarity is higher WITHIN a topic than
ACROSS topics — i.e. semantic basins, not a universal invariant.

  docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
    python -m prayoga.axis_c.run_attractor_basins --model gemma-2-2b-it --hf-id google/gemma-2-2b-it

Aggregate-only output (similarities, no generated text). Claim-tier: MECHANISM (a
measured dynamical property); the turīya mapping stays falsified and is not upgraded.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from prayoga.lm.hf_model import HFModel

# seeds grouped by topic — the basin label is the topic, the test is whether finals
# cluster by topic. Benign content (no dual-use).
TOPICS = {
    "cooking": [
        "Describe how to bake a simple loaf of bread.",
        "Explain how to make a basic tomato soup.",
        "Outline how to brew a cup of tea.",
    ],
    "astronomy": [
        "Describe how stars form in a nebula.",
        "Explain why the moon has phases.",
        "Outline how a telescope gathers light.",
    ],
}
PARAPHRASE = "Rephrase the following text in your own words, preserving its meaning:\n\n{x}"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default="results/axis_c")
    args = ap.parse_args()

    model = HFModel(args.hf_id)
    finals = {}      # seed_id -> final embedding
    topic_of = {}    # seed_id -> topic
    converge = []    # per-seed last-step similarity (per-seed convergence)
    for topic, seeds in TOPICS.items():
        for si, seed in enumerate(seeds):
            text = seed
            prev_emb = model.embed_text(text)
            last_sim = 1.0
            for _ in range(args.iters):
                text = model.generate([PARAPHRASE.format(x=text)], 60)[0].strip() or text
                emb = model.embed_text(text)
                last_sim = _cos(prev_emb, emb)
                prev_emb = emb
            sid = f"{topic}_{si}"
            finals[sid] = prev_emb
            topic_of[sid] = topic
            converge.append(last_sim)

    ids = list(finals)
    within, across = [], []
    for a, b in combinations(ids, 2):
        s = _cos(finals[a], finals[b])
        (within if topic_of[a] == topic_of[b] else across).append(s)

    within_m, across_m = float(np.mean(within)), float(np.mean(across))
    summary = {
        "model": args.model, "iters": args.iters, "n_seeds": len(ids),
        "per_seed_convergence_mean": round(float(np.mean(converge)), 4),
        "within_topic_final_similarity": round(within_m, 4),
        "across_topic_final_similarity": round(across_m, 4),
        "semantic_basin_gap": round(within_m - across_m, 4),
        "verdict": (
            "semantic_basins" if within_m - across_m > 0.05 else "no_topic_structure"
        ),
        "interpretation": (
            "If within-topic >> across-topic, paraphrase dynamics have content-determined "
            "semantic fixed points (multiple basins), NOT a universal content-free turīya. "
            "F7's falsification of the single attractor stands; this characterizes what is real."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"attractor_basins_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ATTRACTOR BASINS (source-of-truth for F7) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
