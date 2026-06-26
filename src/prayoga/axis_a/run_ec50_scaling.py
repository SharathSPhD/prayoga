"""Deep dive (surpasses SOTA): the EC50 pharmacology scaling law for refusal.

SOTA reports jailbreak success as a binary/coarse ASR. prayoga (F2) is first to report
refusal ablation as a continuous *dose-response* with a half-maximal EC50. This extends
that single point into a SCALING LAW: fit EC50 across a model-size series and ask whether
refusal potency scales with model size (the hypothesis: larger models refuse via a
higher-EC50 / more-redundant policy, i.e. they are harder to half-ablate).

For each model: select the extraction layer by max full-ablation ASR on a val split,
sweep partial-ablation strength alpha at that layer, fit a 4-param logistic -> EC50.
Then fit log(EC50) ~ log(params) within and across families.

  docker exec prayoga-gpu python -m prayoga.axis_a.run_ec50_scaling

Claim-tier: MECHANISM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prayoga.axis_a.direction_extraction import directions_all_layers
from prayoga.axis_a.dose_response import fit_dose_response
from prayoga.axis_a.intervention_engine import InterventionEngine
from prayoga.axis_a.refusal import asr
from prayoga.lm.hf_model import HFModel

# (model, hf_id, params_in_billions). Ungated scale series.
SERIES = [
    ("qwen2.5-0.5b-it", "Qwen/Qwen2.5-0.5B-Instruct", 0.5, "Qwen2.5"),
    ("qwen2.5-1.5b-it", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, "Qwen2.5"),
    ("qwen2.5-3b-it", "Qwen/Qwen2.5-3B-Instruct", 3.1, "Qwen2.5"),
    ("qwen2.5-7b-it", "Qwen/Qwen2.5-7B-Instruct", 7.6, "Qwen2.5"),
    ("gemma-2-2b-it", "google/gemma-2-2b-it", 2.6, "Gemma2"),
    ("gemma-2-9b-it", "google/gemma-2-9b-it", 9.2, "Gemma2"),
]


def _load(p: str) -> list[str]:
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def _select_layer(model, eng, dirs, h_val, mnt):
    """Pick the layer whose full ablation maximises ASR on the val split."""
    n = model.n_layers
    cands = sorted(set(int(round(f * (n - 1))) for f in np.linspace(0.25, 0.75, 8)))
    best_l, best_a = cands[0], -1.0
    for L in cands:
        a = asr(eng.ablate_generate(h_val, dirs[L], mnt, alpha=1.0))
        if a > best_a:
            best_a, best_l = a, L
    return best_l, best_a


def main() -> None:
    harmful = _load("data/prompts/harmful.txt")
    harmless = _load("data/prompts/harmless.txt")
    h_tr, h_val, h_ev = harmful[:24], harmful[24:34], harmful[34:]
    alphas = np.linspace(0.0, 1.0, 13).tolist()

    rows = []
    for name, hf_id, params, family in SERIES:
        try:
            model = HFModel(hf_id)
            eng = InterventionEngine(model)
            dirs = directions_all_layers(model, h_tr, harmless[:24])
            L, val_asr = _select_layer(model, eng, dirs, h_val, 24)
            d = dirs[L]
            real = [asr(eng.ablate_generate(h_ev, d, 32, alpha=a)) for a in alphas]
            fit = fit_dose_response(alphas, real)
            rows.append({"model": name, "family": family, "params_B": params, "layer": L,
                         "n_layers": model.n_layers, "ec50": round(fit["ec50"], 4),
                         "slope": round(fit["slope"], 3), "r2": round(fit["r2"], 4),
                         "asr_max": round(real[-1], 3)})
            print(f"[ec50] {name:18s} L={L:2d} EC50={fit['ec50']:.3f} R2={fit['r2']:.3f} asr_max={real[-1]:.2f}")
            del model, eng
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"[skip] {name}: {str(e)[:90]}")

    # scaling-law fits: log(EC50) = log(A) - beta*log(params)
    def fit_law(subset):
        if len(subset) < 3:
            return None
        x = np.log(np.array([r["params_B"] for r in subset]))
        y = np.log(np.array([r["ec50"] for r in subset]))
        beta, logA = np.polyfit(x, y, 1)  # slope, intercept of log(EC50) vs log(params)
        yhat = beta * x + logA
        ss = 1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
        return {"beta": round(float(beta), 3), "A": round(float(np.exp(logA)), 4), "r2": round(float(ss), 3), "n": len(subset)}

    qwen = [r for r in rows if r["family"] == "Qwen2.5"]
    summary = {
        "rows": rows,
        "scaling_law_qwen": fit_law(qwen),
        "scaling_law_all": fit_law(rows),
        "note": "EC50 = A * params_B^beta (beta>0 => larger models need MORE ablation to half-suppress refusal)",
    }
    out = Path("results/axis_a"); out.mkdir(parents=True, exist_ok=True)
    (out / "ec50_scaling.json").write_text(json.dumps(summary, indent=2))
    print("\n=== EC50 SCALING LAW ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
