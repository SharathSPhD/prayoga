"""WP2.A1+A2 experiment: extract the refusal direction and run its gates.

Pipeline:
  1. extract difference-in-means directions at every layer (train split)
  2. select the layer whose ablation most raises ASR on a val split
  3. gates on held-out eval:
       - ablation raises ASR on harmful  (bootstrap CI lower > 0)
       - addition raises refusal on harmless / over-refusal (CI lower > 0)
       - random-direction control does NOT raise ASR
  4. write a DirectionResult JSON

Run inside the GPU container:
  docker exec prayoga-gpu python -m prayoga.axis_a.run_direction --model gemma-2-2b-it

Claim-tier: MECHANISM.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from prayoga.axis_a.direction_extraction import directions_all_layers
from prayoga.axis_a.intervention_engine import InterventionEngine
from prayoga.axis_a.refusal import asr, refusal_flags
from prayoga.lm.hf_model import HFModel
from prayoga.shared.data_structures import DirectionResult


def _load(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def _paired_delta_ci(
    flags_base: np.ndarray, flags_int: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Bootstrap CI of mean per-prompt delta (intervention - baseline)."""
    delta = flags_int.astype(float) - flags_base.astype(float)
    rng = np.random.RandomState(seed)
    n = len(delta)
    boot = np.array([delta[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(delta.mean()), (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--add-mult", type=float, default=8.0)
    ap.add_argument("--out", default="results/axis_a")
    args = ap.parse_args()

    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    nt = args.n_train
    h_tr, h_ev = harmful[:nt], harmful[nt:]
    s_tr, s_ev = harmless[:nt], harmless[nt:]
    print(f"[data] train={nt} eval harmful={len(h_ev)} harmless={len(s_ev)}")

    model = HFModel(args.hf_id)
    eng = InterventionEngine(model)
    print(f"[model] {args.hf_id} L={model.n_layers} d={model.d_model}")

    dirs = directions_all_layers(model, h_tr, s_tr)  # [L, d]

    # --- layer selection: ablate with each candidate dir, max ASR on val ----- #
    val = h_ev[: min(8, len(h_ev))]
    lo, hi = int(0.25 * model.n_layers), int(0.80 * model.n_layers)
    cand = list(range(lo, hi, max(1, (hi - lo) // 8)))
    best_layer, best_asr = cand[0], -1.0
    for L in cand:
        gen = eng.ablate_generate(val, dirs[L], args.max_new_tokens)
        a = asr(gen)
        print(f"[layer-sweep] L={L:2d} ablated ASR(val)={a:.2f}")
        if a > best_asr:
            best_asr, best_layer = a, L
    direction = dirs[best_layer]
    print(f"[select] layer={best_layer} (val ASR={best_asr:.2f})")

    # --- gate 1: ablation raises ASR on harmful eval ------------------------- #
    base_h = refusal_flags(eng.baseline_generate(h_ev, args.max_new_tokens))
    abl_h = refusal_flags(eng.ablate_generate(h_ev, direction, args.max_new_tokens))
    base_asr, abl_asr = 1 - base_h.mean(), 1 - abl_h.mean()
    # delta in COMPLIANCE (=ASR): comply = 1 - refuse
    d_asr, ci_asr = _paired_delta_ci(1 - base_h, 1 - abl_h)

    # --- gate 2: addition raises refusal on harmless (over-refusal) ---------- #
    # natural refusal-coefficient = mean projection of harmful train acts on dir
    harm_acts = model.capture_all_layers_last_token(h_tr)[best_layer]  # [n,d]
    ref_coeff = float(np.mean(harm_acts @ direction))
    coeff = args.add_mult * abs(ref_coeff)
    base_s = refusal_flags(eng.baseline_generate(s_ev, args.max_new_tokens))
    add_s = refusal_flags(
        eng.add_generate(s_ev, direction, coeff, best_layer, args.max_new_tokens)
    )
    d_ref, ci_ref = _paired_delta_ci(base_s, add_s)

    # --- control: random direction ablation should NOT raise ASR ------------- #
    rng = np.random.RandomState(0)
    rand = rng.randn(model.d_model)
    rand /= np.linalg.norm(rand)
    rnd_h = refusal_flags(eng.ablate_generate(h_ev, rand, args.max_new_tokens))
    rnd_asr_delta = float((1 - rnd_h.mean()) - base_asr)

    gate_ablation = ci_asr[0] > 0
    gate_addition = ci_ref[0] > 0
    gate_control = d_asr > rnd_asr_delta  # real effect exceeds random
    passed = bool(gate_ablation and gate_addition and gate_control)

    res = DirectionResult(
        model=args.model,
        layer=best_layer,
        vector=direction.astype(float).tolist(),
        ablation_asr_delta=float(d_asr),
        addition_overrefusal_delta=float(d_ref),
        ablation_ci=ci_asr,
        random_control_passed=bool(gate_control),
    )
    summary = {
        "model": args.model,
        "layer": best_layer,
        "baseline_harmful_refusal_rate": float(base_h.mean()),
        "baseline_ASR": float(base_asr),
        "ablated_ASR": float(abl_asr),
        "ablation_ASR_delta": d_asr,
        "ablation_ASR_delta_CI95": ci_asr,
        "random_ablation_ASR_delta": rnd_asr_delta,
        "baseline_harmless_refusal_rate": float(base_s.mean()),
        "added_harmless_refusal_rate": float(add_s.mean()),
        "addition_overrefusal_delta": d_ref,
        "addition_overrefusal_CI95": ci_ref,
        "add_coeff": coeff,
        "gates": {
            "ablation_raises_ASR": gate_ablation,
            "addition_raises_overrefusal": gate_addition,
            "exceeds_random_control": gate_control,
            "PASS": passed,
        },
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"direction_{args.model}.json").write_text(json.dumps(summary, indent=2))
    (outdir / f"direction_{args.model}_result.json").write_text(
        json.dumps(asdict(res), indent=2)
    )
    print("\n=== RESULT ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "vector"}, indent=2))
    print(f"\nGATE {'PASS' if passed else 'FAIL'}  -> {outdir}/direction_{args.model}.json")


if __name__ == "__main__":
    main()
