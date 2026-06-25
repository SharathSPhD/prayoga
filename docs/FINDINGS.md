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

| Measure | Result |
|---|---|
| Baseline harmful refusal rate | 1.00 (ASR 0.00) |
| **Directional ablation** → harmful ASR | **0.92** (Δ +0.92, bootstrap CI95 [0.79, 1.00]) |
| **Random-direction control** → ASR Δ | **0.00** (no effect) |
| **Activation addition** → harmless refusal rate | 0.04 → **1.00** (over-refusal Δ +0.96, CI95 [0.88, 1.00]) |

**Gates:** ablation-raises-ASR ✓ · addition-raises-over-refusal ✓ ·
exceeds-random-control ✓ → **PASS**.

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
