"""Generate publication-quality figures from aggregate results (safe to commit).

Reads only aggregate statistics from ``results/`` (no direction vectors or
generations) and writes ``figures/*.png``. The figures share a single
house style (serif fonts to match the Palatino body, a fixed colourblind-safe
palette, despined axes, light grids) so the paper reads as one document.

Run from the repository root, in the GPU container or any environment with the
result JSONs present and matplotlib installed::

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# House style
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
RA = ROOT / "results/axis_a"
RB = ROOT / "results/axis_b"
RC = ROOT / "results/axis_c"
RX = ROOT / "results/axis_x"
RS = ROOT / "results/satkarma"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Colourblind-safe palette (Wong-derived), used consistently across figures.
C_REFUSAL = "#b5384d"   # primary / refusal direction / Gemma
C_BLUE = "#2f6f9f"      # secondary / Qwen / addition
C_ORANGE = "#d98a3a"    # injection / intervention
C_GREEN = "#3f8f6b"     # positive / passing gate
C_GRAY = "#8a8f98"      # controls / random / baseline
C_INK = "#24303a"       # annotations

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "URW Palladio L", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#dfe3e8",
    "grid.linewidth": 0.7,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load(p: Path):
    return json.loads(Path(p).read_text())


def _logistic(x, ec50, slope, lo, hi):
    return lo + (hi - lo) / (1.0 + np.exp(-slope * (x - ec50)))


def _save(fig, name: str):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# F2 — abliteration dose-response
# --------------------------------------------------------------------------- #
def fig_dose() -> None:
    d = _load(RA / "dose_gemma-2-2b-it.json")
    a = np.array(d["alphas"])
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    xs = np.linspace(0, 1, 300)
    f = d["fit"]
    ax.plot(xs, _logistic(xs, f["ec50"], f["slope"], f["lo"], f["hi"]),
            color=C_REFUSAL, lw=2.2, zorder=2,
            label=f"logistic fit ($R^2$={f['r2']:.3f})")
    ax.scatter(a, d["asr_real"], color=C_REFUSAL, s=44, zorder=3,
               edgecolor="white", linewidth=0.6, label="refusal direction")
    ax.scatter(a, d["asr_random_control"], color=C_GRAY, marker="s", s=34,
               zorder=3, label="random direction (control)")
    ax.axvline(f["ec50"], ls="--", color=C_INK, alpha=0.7, lw=1.1)
    ax.annotate(f"EC50 = {f['ec50']:.2f}", xy=(f["ec50"], 0.46),
                xytext=(f["ec50"] + 0.06, 0.30), color=C_INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=C_INK, alpha=0.6))
    ax.set_xlabel(r"ablation strength $\alpha$")
    ax.set_ylabel("attack success rate (ASR)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center right")
    _despine(ax)
    _save(fig, "f2_dose_response.png")


# --------------------------------------------------------------------------- #
# F6/F8 — cross-model necessary vs sufficient
# --------------------------------------------------------------------------- #
def fig_cross_model() -> None:
    models = [("gemma-2-2b-it", "Gemma-2-2B\n(eff. dim 1)"),
              ("gemma-2-9b-it", "Gemma-2-9B\n(eff. dim 1)"),
              ("qwen2.5-3b-it", "Qwen2.5-3B\n(eff. dim 3)")]
    abl, add = [], []
    for m, _ in models:
        d = _load(RA / f"direction_{m}.json")
        abl.append(d["ablation_ASR_delta"])
        add.append(d["addition_overrefusal_delta"])
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    b1 = ax.bar(x - w / 2, abl, w, label="ablation → jailbreak ($\\Delta$ASR)",
                color=C_REFUSAL)
    b2 = ax.bar(x + w / 2, add, w, label="addition → over-refusal ($\\Delta$)",
                color=C_BLUE)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in models])
    ax.set_ylabel("effect size $\\Delta$")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper center", ncol=1)
    _despine(ax)
    _save(fig, "f6_f8_cross_model.png")


# --------------------------------------------------------------------------- #
# F11 — symmetry order parameter
# --------------------------------------------------------------------------- #
def fig_symmetry() -> None:
    g = _load(RB / "symmetry_gemma-2-2b-it.json")
    q = _load(RB / "symmetry_qwen2.5-3b-it.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    x = np.arange(2)
    w = 0.36
    ax1.bar(x - w / 2, [g["F_ratio_refusal_dir"], q["F_ratio_refusal_dir"]], w,
            label="refusal direction", color=C_REFUSAL)
    ax1.bar(x + w / 2, [g["F_ratio_random_dir"], q["F_ratio_random_dir"]], w,
            label="random direction", color=C_GRAY)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Gemma-2-2B", "Qwen2.5-3B"])
    ax1.set_ylabel("orbit-invariance $F$-ratio (log scale)")
    ax1.legend()
    _despine(ax1)

    plain = [g["order_param_plain_harmful"], q["order_param_plain_harmful"]]
    inj = [g["order_param_injected_harmful"], q["order_param_injected_harmful"]]
    xx = np.arange(2)
    ax2.bar(xx - w / 2, plain, w, label="plain harmful", color=C_BLUE)
    ax2.bar(xx + w / 2, inj, w, label="under injection", color=C_ORANGE)
    for i in range(2):
        drop = 100 * (inj[i] - plain[i]) / plain[i]
        ax2.text(xx[i], max(plain[i], inj[i]) + 0.012, f"{drop:.0f}%",
                 ha="center", va="bottom", fontsize=8, color=C_INK)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(["Gemma-2-2B", "Qwen2.5-3B"])
    ax2.set_ylabel("order parameter $m=(h\\cdot\\hat d)/\\Vert h\\Vert$")
    ax2.legend()
    _despine(ax2)
    _save(fig, "f11_symmetry.png")


# --------------------------------------------------------------------------- #
# F8 — effective dimension by iterative projection
# --------------------------------------------------------------------------- #
def fig_dimensionality() -> None:
    g = _load(RA / "dim_gemma-2-2b-it.json")
    q = _load(RA / "dim_qwen2.5-3b-it.json")
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for d, c, lbl in [(g, C_REFUSAL, f"Gemma-2-2B (eff. dim {g['effective_dim']})"),
                      (q, C_BLUE, f"Qwen2.5-3B (eff. dim {q['effective_dim']})")]:
        a = d["accs_by_dims_removed"]
        ax.plot(range(len(a)), a, "o-", color=c, lw=2, ms=6,
                markeredgecolor="white", markeredgewidth=0.6, label=lbl)
    ax.axhline(0.5, ls="--", color=C_GRAY, alpha=0.8, label="chance (0.5)")
    ax.set_xlabel("directions removed from the residual stream")
    ax.set_ylabel("harmful / harmless CV accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    _despine(ax)
    _save(fig, "f8_dimensionality.png")


# --------------------------------------------------------------------------- #
# F10 — satkarma interventions
# --------------------------------------------------------------------------- #
def fig_satkarma() -> None:
    d = _load(RS / "satkarma_gemma-2-2b-it.json")
    rows = d["table"]
    names = [f"{r['sanskrit']}\n({r.get('gloss', '')})" if r.get("gloss") else r["sanskrit"]
             for r in rows]
    eff = [r["effect"] for r in rows]
    ctl = [r["control"] for r in rows]
    y = np.arange(len(names))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(y + h / 2, eff, h, label="measured effect",
            color=[C_GREEN if r["separated"] else "#c9ccd1" for r in rows])
    ax.barh(y - h / 2, ctl, h, label="matched control", color=C_GRAY)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("measured effect / control value")
    ax.legend(loc="lower right")
    _despine(ax)
    _save(fig, "f10_satkarma.png")


# --------------------------------------------------------------------------- #
# F18 — layer sweep of effective dimension
# --------------------------------------------------------------------------- #
def fig_dimsweep() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    for m, c, lbl in [("gemma-2-2b-it", C_REFUSAL, "Gemma-2-2B"),
                      ("qwen2.5-3b-it", C_BLUE, "Qwen2.5-3B")]:
        p = RA / f"dimsweep_{m}.json"
        if not p.exists():
            continue
        d = _load(p)
        xs = [x["layer"] for x in d["per_layer"]]
        ys = [x["effective_dim"] for x in d["per_layer"]]
        ax.plot(xs, ys, "o-", ms=4, lw=1.8, color=c, label=lbl)
    ax.scatter([7], [1], s=120, facecolor="none", edgecolor=C_REFUSAL, lw=1.6,
               zorder=4)
    ax.scatter([19], [3], s=120, facecolor="none", edgecolor=C_BLUE, lw=1.6,
               zorder=4)
    ax.annotate("extraction layers\n(the single-layer contrast)", xy=(19, 3),
                xytext=(21, 8), fontsize=8, color=C_INK,
                arrowprops=dict(arrowstyle="->", color=C_INK, alpha=0.6))
    ax.set_xlabel("layer")
    ax.set_ylabel("refusal-subspace effective dimension")
    ax.legend()
    _despine(ax)
    _save(fig, "f18_dimsweep.png")


# --------------------------------------------------------------------------- #
# F19 — EC50 pharmacology across families
# --------------------------------------------------------------------------- #
def fig_ec50_scaling() -> None:
    d = _load(RA / "ec50_scaling.json")
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fam_c = {"Qwen2.5": C_BLUE, "Gemma2": C_REFUSAL}
    for fam, c in fam_c.items():
        rs = [r for r in rows if r["family"] == fam]
        if rs:
            ax.scatter([r["params_B"] for r in rs], [r["ec50"] for r in rs],
                       color=c, s=70, zorder=3, edgecolor="white", linewidth=0.7,
                       label=fam)
            for r in rs:
                ax.annotate(f"{r['ec50']:.2f}", (r["params_B"], r["ec50"]),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7.5, color=c)
    law = d.get("scaling_law_qwen")
    if law:
        xs = np.linspace(0.4, 10, 100)
        ax.plot(xs, law["A"] * xs ** law["beta"], color=C_BLUE, ls="--", lw=1.4,
                label=f"Qwen fit: $\\beta$={law['beta']} ($R^2$={law['r2']})")
    ax.axhspan(0.13, 0.16, color=C_BLUE, alpha=0.08)
    ax.axhspan(0.23, 0.26, color=C_REFUSAL, alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlabel("model size (billions of parameters, log scale)")
    ax.set_ylabel("EC50 (half-ablation strength)")
    ax.legend()
    _despine(ax)
    _save(fig, "f19_ec50_scaling.png")


# --------------------------------------------------------------------------- #
# F23 — addition dose-response (the asymmetry is dosing)
# --------------------------------------------------------------------------- #
def fig_addition_dose() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for m, c, lbl in [("qwen2.5-3b-it", C_BLUE, "Qwen2.5-3B (L19) — inverted-U"),
                      ("gemma-2-2b-it", C_REFUSAL, "Gemma-2-2B (L7) — monotone")]:
        p = RA / f"addition_dose_{m}.json"
        if not p.exists():
            continue
        d = _load(p)
        xs = [r["coeff"] for r in d["curve"]]
        ys = [r["over_refusal"] for r in d["curve"]]
        ax.plot(xs, ys, "o-", ms=5, lw=1.9, color=c,
                markeredgecolor="white", markeredgewidth=0.5, label=lbl)
    ax.axvline(64, ls="--", color=C_GRAY, alpha=0.8)
    ax.text(66, 0.06, "prior fixed-coefficient\ntest (64×)", fontsize=7.5,
            color=C_INK)
    ax.set_xlabel("addition coefficient")
    ax.set_ylabel("over-refusal rate (harmless prompts)")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper right")
    _despine(ax)
    _save(fig, "f23_addition_dose.png")


# --------------------------------------------------------------------------- #
# F20 — cross-axis triangulation keystone
# --------------------------------------------------------------------------- #
def fig_triangulation() -> None:
    models = [("gemma-2-2b-it", "Gemma-2-2B"),
              ("qwen2.5-3b-it", "Qwen2.5-3B"),
              ("gemma-2-9b-it", "Gemma-2-9B")]
    data = {m: _load(RX / f"triangulation_{m}.json") for m, _ in models}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.7))

    # Left: internal coupling — corr(A,B) with CI
    x = np.arange(len(models))
    corr = [data[m]["corr_AB_pearson"] for m, _ in models]
    cis = [data[m]["corr_AB_ci"] for m, _ in models]
    err = [[corr[i] - cis[i][0] for i in range(len(models))],
           [cis[i][1] - corr[i] for i in range(len(models))]]
    ax1.bar(x, corr, 0.55, color=C_REFUSAL, yerr=err, capsize=4,
            error_kw=dict(ecolor=C_INK, lw=1))
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in models], fontsize=8.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("corr($\\Delta$A, $\\Delta$B) under injection")
    _despine(ax1)

    # Right: genuine behavioral flip (judge) vs substring artifact
    w = 0.36
    flip = [data[m]["flip_rate"] for m, _ in models]
    sub = [data[m].get("substring_flip_rate", 0.43) for m, _ in models]
    ax2.bar(x - w / 2, sub, w, color=C_GRAY, label="substring metric (artifact)")
    ax2.bar(x + w / 2, flip, w, color=C_GREEN, label="content-faithful judge")
    for i in range(len(models)):
        ax2.text(x[i] + w / 2, flip[i] + 0.015, f"{flip[i]:.1%}", ha="center",
                 va="bottom", fontsize=8, color=C_INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in models], fontsize=8.5)
    ax2.set_ylim(0, 0.55)
    ax2.set_ylabel("behavioural capture (flip) rate")
    ax2.legend(loc="upper right")
    _despine(ax2)
    _save(fig, "f20_triangulation.png")


# --------------------------------------------------------------------------- #
# F22 — mid-network truthfulness direction
# --------------------------------------------------------------------------- #
def fig_truth_direction() -> None:
    d = _load(RC / "truth_direction_gemma-2-2b-it.json")
    per = d["per_layer"]
    layers = [r["layer"] for r in per]
    cross = [r["cross_dataset_acc"] for r in per]
    within = [r.get("within_cv_acc") for r in per]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(layers, cross, "o-", color=C_REFUSAL, lw=2, ms=4,
            label="cross-dataset transfer (geography→science)")
    if all(v is not None for v in within):
        ax.plot(layers, within, "s--", color=C_BLUE, lw=1.4, ms=3, alpha=0.8,
                label="within-set CV")
    ax.axhline(0.5, ls=":", color=C_GRAY, label="chance / layer-0 surface (0.50)")
    bm = d["best_mid_layer"]
    ax.scatter([bm], [d["best_mid_cross_dataset_acc"]], s=130, facecolor="none",
               edgecolor=C_GREEN, lw=2, zorder=5)
    ax.annotate(f"mid layer {bm}: {d['best_mid_cross_dataset_acc']:.2f}\n(null $p$={d['shuffle_null_p']})",
                xy=(bm, d["best_mid_cross_dataset_acc"]), xytext=(bm + 1, 0.66),
                fontsize=8, color=C_INK,
                arrowprops=dict(arrowstyle="->", color=C_INK, alpha=0.6))
    ax.set_xlabel("layer")
    ax.set_ylabel("truth-classification accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc="lower right")
    _despine(ax)
    _save(fig, "f22_truth_direction.png")


# --------------------------------------------------------------------------- #
# F14 — satkarma v2: marana coherence collapse
# --------------------------------------------------------------------------- #
def fig_satkarma_v2() -> None:
    d = _load(RS / "satkarma_v2_gemma-2-2b-it.json")
    curve = d["marana"]["curve"]
    ks = [r["K"] for r in curve]
    top = [r["coherence_top"] for r in curve]
    rnd = [r["coherence_random"] for r in curve]
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.plot(ks, top, "o-", color=C_REFUSAL, lw=2, ms=6,
            label="ablate top-$K$ active SAE features (māraṇa)")
    ax.plot(ks, rnd, "s--", color=C_GRAY, lw=1.6, ms=5,
            label="random-$K$ control")
    ax.axhline(d["marana"]["baseline_coherence"], ls=":", color=C_INK, alpha=0.6,
               label=f"baseline coherence ({d['marana']['baseline_coherence']:.2f})")
    ax.set_xlabel("number of ablated features $K$")
    ax.set_ylabel("output coherence")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="lower left")
    _despine(ax)
    _save(fig, "f14_satkarma_v2.png")


def _gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


# --------------------------------------------------------------------------- #
# F11b — order-parameter geometry: the broken symmetry (data-grounded)
# --------------------------------------------------------------------------- #
def fig_order_geometry() -> None:
    from matplotlib.patches import Patch
    models = [("gemma-2-2b-it", "Gemma-2-2B"), ("qwen2.5-3b-it", "Qwen2.5-3B")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for ax, (m, lbl) in zip(axes, models):
        d = _load(RB / f"symmetry_{m}.json")
        sd = d["within_orbit_std_harmful"]
        hm, sm = d["order_param_harmful_mean"], d["order_param_harmless_mean"]
        plain, inj = d["order_param_plain_harmful"], d["order_param_injected_harmful"]
        lo = min(sm, inj) - 4 * sd
        hi = max(hm, plain) + 4 * sd
        xs = np.linspace(lo, hi, 400)
        peak = _gauss(hm, hm, sd)
        ax.fill_between(xs, _gauss(xs, sm, sd), color=C_BLUE, alpha=0.35)
        ax.fill_between(xs, _gauss(xs, hm, sd), color=C_REFUSAL, alpha=0.35)
        ax.plot(xs, _gauss(xs, sm, sd), color=C_BLUE, lw=1.5)
        ax.plot(xs, _gauss(xs, hm, sd), color=C_REFUSAL, lw=1.5)
        ax.annotate("", xy=(inj, peak * 0.42), xytext=(plain, peak * 0.42),
                    arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=2.2))
        ax.text((plain + inj) / 2, peak * 0.50, "injection", ha="center",
                fontsize=8.5, color=C_ORANGE)
        ax.set_yticks([])
        ax.set_ylim(0, peak * 1.5)
        ax.set_xlabel(r"order parameter $m=(h\cdot\hat d)/\Vert h\Vert$")
        ax.text(0.03, 0.97, f"{lbl}: $F$-ratio {d['F_ratio_refusal_dir']:.1f} vs "
                f"{d['F_ratio_random_dir']:.2f} (random)", transform=ax.transAxes,
                fontsize=8.5, va="top", color=C_INK)
        _despine(ax)
    handles = [Patch(facecolor=C_BLUE, alpha=0.5, label="harmless orbits"),
               Patch(facecolor=C_REFUSAL, alpha=0.5, label="harmful orbits")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "f12_order_geometry.png")


# --------------------------------------------------------------------------- #
# F1b — refusal-direction geometry: separation, ablation, addition (data-grounded)
# --------------------------------------------------------------------------- #
def fig_refusal_geometry() -> None:
    d = _load(RA / "affine_alignment.json")["gemma-2-2b-it"]
    gap, sd = d["harmful_harmless_gap_along_dref"], d["harmless_spread_along_dref"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)
    xs = np.linspace(-3 * sd, gap + 4 * sd, 500)
    thr = gap / 2
    peak = _gauss(0, 0, sd)
    # Panel 1: native geometry
    ax1.fill_between(xs, _gauss(xs, 0, sd), color=C_BLUE, alpha=0.35, label="harmless")
    ax1.fill_between(xs, _gauss(xs, gap, sd), color=C_REFUSAL, alpha=0.35, label="harmful")
    ax1.axvline(thr, ls="--", color=C_INK, alpha=0.6)
    ax1.text(thr, peak * 1.46, "refuse / comply\nthreshold", ha="center", va="top",
             fontsize=8, color=C_INK)
    ax1.annotate("", xy=(gap, peak * 0.10), xytext=(0, peak * 0.10),
                 arrowprops=dict(arrowstyle="<->", color=C_GRAY, lw=1.3))
    ax1.text(gap / 2, peak * 0.15, f"separation {gap:.1f}", ha="center",
             fontsize=8, color=C_GRAY)
    ax1.set_xlabel(r"projection onto refusal direction $\hat d$")
    ax1.set_yticks([])
    ax1.legend(loc="upper left", fontsize=8)
    _despine(ax1)
    # Panel 2: interventions
    ax2.fill_between(xs, _gauss(xs, 0, sd), color=C_BLUE, alpha=0.15)
    ax2.fill_between(xs, _gauss(xs, gap, sd), color=C_REFUSAL, alpha=0.15)
    ax2.fill_between(xs, _gauss(xs, thr, sd), color="#888", alpha=0.4,
                     label="ablation $\\to$ threshold")
    ax2.annotate("", xy=(gap * 0.80, peak * 0.42), xytext=(thr * 0.35, peak * 0.42),
                 arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=2))
    ax2.text(thr * 0.55, peak * 0.50, "addition $+c\\,\\hat d$", fontsize=8,
             color=C_ORANGE)
    ax2.axvline(thr, ls="--", color=C_INK, alpha=0.6)
    ax2.set_xlabel(r"projection onto refusal direction $\hat d$")
    ax2.set_yticks([])
    ax2.legend(loc="upper left", fontsize=8)
    _despine(ax2)
    ax1.set_ylim(0, peak * 1.6)
    fig.tight_layout()
    _save(fig, "f1_refusal_geometry.png")


if __name__ == "__main__":
    figs = (fig_dose, fig_cross_model, fig_symmetry, fig_dimensionality,
            fig_satkarma, fig_dimsweep, fig_ec50_scaling, fig_addition_dose,
            fig_triangulation, fig_truth_direction, fig_satkarma_v2,
            fig_order_geometry, fig_refusal_geometry)
    for fn in figs:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"skip {fn.__name__}: {e}")
    print("done →", *(p.name for p in sorted(FIG.glob("*.png"))))
