# prayoga — Findings ledger

Running log of validated results, each tagged with its claim-tier and the gate
it passed. Raw artifacts (direction vectors, generations) are **safety-gated**:
`results/` is git-ignored; only aggregate statistics appear here or in the repo.

---

## F1 — Refusal is mediated by a single direction in Gemma-2-2b-it *(MECHANISM)*

**Date:** 2026-06-25 · **WP:** 2.A1 + 2.A2 · **Model:** `google/gemma-2-2b-it`
(26 layers, d=2304), GB10 GPU. **Method:** difference-in-means direction (Arditi
et al. 2024) on 24 harmful / 24 harmless prompts; layer selected by max ablation
ASR on a val split → **layer 7**. Eval on 24 held-out harmful / 24 harmless.

Canonical numbers are the **corrected** run (disjoint train/val/eval splits;
10-direction random control); the original leaky-split run is shown for contrast.

| Measure | Original (leaky) | **Corrected (canonical)** |
|---|---|---|
| Baseline harmful refusal rate | 1.00 (ASR 0.00) | 1.00 (ASR 0.00) |
| **Directional ablation** → harmful ASR | 0.92 (CI [0.79,1.0]) | **0.90** (Δ+0.90, CI95 [0.75, 1.00]) |
| **Random-direction control** → ASR Δ | 0.00 (1 dir) | **0.00** (max & mean over 10 dirs) |
| **Activation addition** → harmless refusal | 0.04 → 1.00 (Δ+0.96) | 0.05 → **1.00** (Δ+0.95, CI95 [0.85, 1.00]) |

**Gates (corrected):** ablation-raises-ASR ✓ · addition-raises-over-refusal ✓ ·
exceeds-random-control (10-dir max) ✓ → **PASS**.

**Robustness to the review:** the adversarial review's falsification test was
*"if fixing the layer-selection leakage + weak control drops the effect >50%, it
was an artifact."* Fixing both moved ablation ASR 0.92 → 0.90 (a 2-point change)
— so the effect is **real, not leakage-inflated**.

**Adversarial validation:** ablated generations were inspected and produce
*genuine* compliant harmful content (not mere omission of refusal phrases);
baseline genuinely refuses. The substring ASR metric is therefore not inflating
the effect for this model.

**Interpretation (tier-disciplined):** this is the MECHANISM-tier linchpin of the
program — refusal in this model is a low-dimensional, causally-efficacious
residual-stream direction that can be measured, ablated (suppressing refusal),
and added (inducing over-refusal). It is the computational correlate the symmetry
thesis (MO-1) rests on. It says nothing yet about ANALOGY (Axis B) or METAPHOR
(Axis C) — those are separate gates.

**Open follow-ups:** A-3 dimensionality (single vs affine, Marshall 2411.09003);
A-5 dose-response/EC50; scale sweep (1B→4B→9B); transfer across models.

**Dual-use:** the layer-7 direction is an abliteration vector; it is NOT released
in the public repo (kept under git-ignored `results/`). Any future release is
safety-gated per the program's responsible-disclosure commitment.

---

## F2 — Ablation dose-response and EC50 in Gemma-2-2b-it *(MECHANISM)*

**Date:** 2026-06-25 · **WP:** 2.A5 · **Model:** `google/gemma-2-2b-it`, layer 7.
**Method:** partial directional ablation at strength α∈[0,1] (h ↦ h − α·(h·d̂)d̂
everywhere); ASR on 24 held-out harmful prompts vs α; 4-param logistic fit. A
random direction is swept as a flat control.

| α | 0.0 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|---|
| ASR (real) | 0.00 | 0.12 | 0.38 | 0.58 | 0.71 | 0.88 | 0.92 |
| ASR (random) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Fit:** **EC50 = 0.329**, slope = 9.76, **R² = 0.996**. Gates: monotone-rise ✓ ·
EC50∈(0,1) ✓ · random-control-flat ✓ → **PASS**.

**Interpretation (MECHANISM):** refusal suppression is a smooth, monotone,
*dosable* function of how much of the single direction is removed — the
quantitative "vaśīkaraṇa-as-dose-response" core. The random control is flat at
every dose, so the curve is specific to the refusal direction, not a generic
perturbation magnitude. EC50≈0.33 means removing ~⅓ of the direction's
projection already half-collapses refusal.
