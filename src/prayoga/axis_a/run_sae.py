"""WP2.A4 experiment: train a BatchTopK SAE and find causal refusal features.

Builds a real activation corpus from a text dataset, trains the SAE, identifies
features that fire more on harmful than harmless prompts, and validates causality
by ablating the top refusal feature's decoder direction during generation
(expecting ASR to rise above baseline and above a random-feature control).

  docker exec prayoga-gpu python -m prayoga.axis_a.run_sae --model gemma-2-2b-it --layer 7

Claim-tier: MECHANISM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.intervention_engine import InterventionEngine
from prayoga.axis_a.refusal import asr
from prayoga.axis_a.sae import feature_activations, train_sae
from prayoga.lm.hf_model import HFModel


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def _corpus_texts(n: int) -> list[str]:
    """Real text for the SAE corpus (wikitext-2); fall back to project prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 40][:n]
        if texts:
            return texts
    except Exception as e:  # network/dep failure -> honest fallback
        print(f"[corpus] dataset unavailable ({str(e)[:60]}); using project prompts")
    base = _load("data/prompts/harmful.txt") + _load("data/prompts/harmless.txt") \
        + _load("data/prompts/regimes/jagrat.txt") + _load("data/prompts/regimes/svapna.txt")
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--n-features", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--corpus-seqs", type=int, default=500)
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    model = HFModel(args.hf_id)
    eng = InterventionEngine(model)
    print(f"[sae] collecting corpus activations at layer {args.layer} ...")
    corpus = model.capture_positions(_corpus_texts(args.corpus_seqs), args.layer)
    print(f"[sae] corpus = {corpus.shape}; training SAE F={args.n_features} k={args.k}")
    sae, metrics = train_sae(corpus, args.n_features, args.k, steps=args.steps)
    print(f"[sae] trained: FVU={metrics['fvu']:.3f} L0={metrics['l0']:.1f}")

    # refusal features: fire more on harmful than harmless (last-token, chat-templated)
    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    Xh = model.capture_all_layers_last_token(harmful)[args.layer]
    Xs = model.capture_all_layers_last_token(harmless)[args.layer]
    zh = feature_activations(sae, Xh).mean(0)
    zs = feature_activations(sae, Xs).mean(0)
    gap = zh - zs
    top = int(np.argmax(gap))
    rng = np.random.RandomState(0)
    rand_feat = int(rng.randint(0, args.n_features))
    dec = sae._unit_dec().detach().cpu().numpy()  # [F, d]

    # causal validation: ablate the refusal feature's decoder direction during gen
    h_ev = harmful[24:]
    base = asr(eng.baseline_generate(h_ev, 40))
    abl_refusal = asr(eng.ablate_generate(h_ev, dec[top], 40))
    abl_random = asr(eng.ablate_generate(h_ev, dec[rand_feat], 40))

    summary = {
        "model": args.model, "layer": args.layer,
        "corpus_tokens": int(corpus.shape[0]), "n_features": args.n_features, "k": args.k,
        "FVU": round(metrics["fvu"], 4), "L0": round(metrics["l0"], 2),
        "top_refusal_feature": top, "refusal_feature_gap": round(float(gap[top]), 4),
        "baseline_ASR": round(base, 3),
        "ASR_ablate_refusal_feature": round(abl_refusal, 3),
        "ASR_ablate_random_feature": round(abl_random, 3),
        "refusal_feature_is_causal": bool(abl_refusal > base + 0.15 and abl_refusal > abl_random + 0.1),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"sae_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SAE REFUSAL FEATURE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
