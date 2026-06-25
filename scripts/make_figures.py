"""Generate publication-quality figures from aggregate results (safe to commit).

Reads only aggregate statistics from results/ (no direction vectors / generations)
and writes figures/*.png. Run in the GPU container (has the result JSONs mounted):

  docker exec -w /workspace/prayoga prayoga-gpu python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results/axis_a")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)


def _logistic(x, ec50, slope, lo, hi):
    return lo + (hi - lo) / (1.0 + np.exp(-slope * (x - ec50)))


def fig_dose() -> None:
    d = json.loads((R / "dose_gemma-2-2b-it.json").read_text())
    a = np.array(d["alphas"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(a, d["asr_real"], color="#c0392b", zorder=3, label="refusal direction")
    xs = np.linspace(0, 1, 200)
    f = d["fit"]
    ax.plot(xs, _logistic(xs, f["ec50"], f["slope"], f["lo"], f["hi"]),
            color="#c0392b", lw=2, label=f"logistic fit (R²={f['r2']:.3f})")
    ax.scatter(a, d["asr_random_control"], color="#7f8c8d", marker="s", label="random direction")
    ax.axvline(f["ec50"], ls="--", color="#2c3e50", alpha=.7)
    ax.text(f["ec50"] + .02, .1, f"EC50={f['ec50']:.2f}", color="#2c3e50")
    ax.set_xlabel("ablation strength α"); ax.set_ylabel("attack success rate (ASR)")
    ax.set_title("Refusal abliteration dose–response (Gemma-2-2b, L7)")
    ax.legend(fontsize=8, loc="center right"); ax.set_ylim(-.05, 1.05)
    fig.tight_layout(); fig.savefig(FIG / "f2_dose_response.png", dpi=160); plt.close(fig)


def fig_cross_model() -> None:
    models = [("gemma-2-2b-it", "Gemma-2-2b\n(dim 1)"), ("gemma-2-9b-it", "Gemma-2-9b\n(dim 1)"),
              ("qwen2.5-3b-it", "Qwen2.5-3b\n(dim 3)")]
    abl, add = [], []
    for m, _ in models:
        d = json.loads((R / f"direction_{m}.json").read_text())
        abl.append(d["ablation_ASR_delta"]); add.append(d["addition_overrefusal_delta"])
    x = np.arange(len(models)); w = .36
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - w / 2, abl, w, label="ablation → jailbreak (ASR Δ)", color="#c0392b")
    ax.bar(x + w / 2, add, w, label="addition → over-refusal (Δ)", color="#2980b9")
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in models], fontsize=8)
    ax.set_ylabel("effect size (Δ)"); ax.set_ylim(0, 1.1)
    ax.set_title("Necessary-vs-sufficient across models")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "f6_f8_cross_model.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    fig_dose()
    fig_cross_model()
    print("wrote", *[p.name for p in FIG.glob("*.png")])
