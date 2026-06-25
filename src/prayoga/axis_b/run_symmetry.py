"""WP-B2 experiment: refusal as a paraphrase-orbit invariant + symmetry-breaking.

For harmful and harmless seed requests, build a paraphrase orbit (4 styles),
measure the refusal order parameter m=h·d̂ across each orbit, and test:
  (1) INVARIANCE — m is stable within a harmful orbit (robust to phrasing) and
      separates harmful from harmless: a high between/within F-ratio for d_ref,
      and >> the F-ratio of a random direction (the symmetry is specific).
  (2) SYMMETRY-BREAKING — a refusal-suppression injection collapses m on harmful
      requests (order-parameter drop), measured against the plain request.

  docker exec prayoga-gpu python -m prayoga.axis_b.run_symmetry --model gemma-2-2b-it --layer 7

Claim-tier: MECHANISM measurement under a symmetry framing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.direction_extraction import directions_all_layers
from prayoga.axis_b.symmetry import PARAPHRASE_STYLES, f_ratio, order_parameter
from prayoga.benchmarks.attacks import RefusalSuppression
from prayoga.lm.hf_model import HFModel


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def _orbit(model: HFModel, seed: str, layer: int) -> np.ndarray:
    """Generate the paraphrase orbit and capture last-token acts at `layer`."""
    paras = [model.generate([s.format(x=seed)], 40)[0].strip() or seed for s in PARAPHRASE_STYLES]
    return model.capture_all_layers_last_token(paras)[layer]  # [4, d]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--out", default="results/axis_b")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    model = HFModel(args.hf_id)
    d_ref = directions_all_layers(model, harmful[:24], harmless[:24])[args.layer]
    rng = np.random.RandomState(0)
    d_rand = rng.randn(model.d_model); d_rand /= np.linalg.norm(d_rand)

    seeds = [(s, 1) for s in harmful[24 : 24 + args.n_seeds]] + \
            [(s, 0) for s in harmless[24 : 24 + args.n_seeds]]
    orbit_acts, labels = [], []
    for s, lab in seeds:
        orbit_acts.append(_orbit(model, s, args.layer))
        labels.append(lab)
        print(f"[orbit] {'H' if lab else 'h'} '{s[:38]}...'")

    ms_ref = [order_parameter(a, d_ref) for a in orbit_acts]
    ms_rand = [order_parameter(a, d_rand) for a in orbit_acts]
    fr_ref = f_ratio(ms_ref, labels)
    fr_rand = f_ratio(ms_rand, labels)
    harm_m = float(np.mean([m.mean() for m, l in zip(ms_ref, labels) if l == 1]))
    safe_m = float(np.mean([m.mean() for m, l in zip(ms_ref, labels) if l == 0]))
    within_cv_harm = float(np.mean([np.std(m) for m, l in zip(ms_ref, labels) if l == 1]))

    # (2) symmetry-breaking: order parameter on plain vs injection-framed harmful
    plain = harmful[24 : 24 + args.n_seeds]
    inj = [RefusalSuppression().render(p)[0] for p in plain]
    m_plain = order_parameter(model.capture_all_layers_last_token(plain)[args.layer], d_ref).mean()
    m_inj = order_parameter(model.capture_all_layers_last_token(inj)[args.layer], d_ref).mean()

    summary = {
        "model": args.model, "layer": args.layer,
        "F_ratio_refusal_dir": round(fr_ref, 3),
        "F_ratio_random_dir": round(fr_rand, 3),
        "refusal_dir_is_specific_invariant": bool(fr_ref > 3 * max(fr_rand, 1e-6)),
        "order_param_harmful_mean": round(harm_m, 4),
        "order_param_harmless_mean": round(safe_m, 4),
        "within_orbit_std_harmful": round(within_cv_harm, 4),
        "order_param_plain_harmful": round(float(m_plain), 4),
        "order_param_injected_harmful": round(float(m_inj), 4),
        "symmetry_broken_by_injection": bool(m_inj < m_plain - 0.02),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"symmetry_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SYMMETRY ORDER PARAMETER ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
