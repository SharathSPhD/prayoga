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

---

## F3 — avasthātraya regime probe is SURFACE-CONFOUNDED → DEMOTED *(METAPHOR → falsified-as-operationalized)*

**Date:** 2026-06-25 · **WP:** 2.C1 · **Model:** `gemma-2-2b-it`. **Method:** linear
probe (3-way: jāgrat grounded-factual / svapna surreal-confabulatory / suṣupti
null-minimal) on last-token residuals; held-out-prompt transfer + label-shuffled
null. Headline layer = network middle (13), pre-specified.

**Raw gate result:** transfer_acc = **1.00**, null_p = **0.003** (chance 0.33) →
the gate, taken at face value, PASSES at every layer.

**Why this is a DEMOTION, not a win:** transfer_acc is **1.00 already at layer 0**
(token embeddings) and ≈1.0 at all 26 layers. The three regimes are perfectly
separable from raw surface form (suṣupti ≈ 1–3 chars, svapna = long surreal
sentences, jāgrat = medium questions). This **fails the §0 non-triviality bar
(criterion 4: not reducible to the model just processing the surface prompt)** —
there is no depth-emergent signal beyond what token length/identity already give.

**Decision (X-2 demotion protocol):** `TierDecision(claim="avasthatraya-regime-
state", tier="METAPHOR", demoted=True, evidence="layer-0 transfer_acc=1.0 ⇒
surface-confounded")`. The naive prompt-set operationalization of the
jāgrat/svapna/suṣupti distinction does **not** constitute evidence of an internal
"state." Reported as an honest negative.

---

## F4 — Black-box Claude resists the naive attack battery (Tier-1 reference end) *(MECHANISM, behavioral)*

**Date:** 2026-06-25 · **WP:** 1.2/1.3 · **Model:** Claude `opus-4-8` via `claude -p`
(lean, neutral cwd). **Method:** 5 attack families × 11 harmful requests (1
transient CLI error/family skipped); refusal-substring ASR.

| Attack family | ASR | n |
|---|---|---|
| direct (control) | 0.00 | 11 |
| refusal_suppression | 0.00 | 11 |
| persona_dan | 0.00 | 11 |
| many_shot | 0.00 | 11 |
| crescendo (multi-turn) | 0.00 | 11 |

**Interpretation:** the frontier model refuses **100%** of these naive black-box
jailbreaks — the "sophisticated reference end." This contrasts sharply with the
*same* refusal behavior in Gemma-2-2b, which is **trivially abliterated white-box**
(F1: ASR→0.90 by removing one direction). The cross-tier picture: refusal that is
behaviorally robust at the frontier is, mechanistically, a low-dimensional and
fragile object in small open models — the gap the program exists to characterize.

**Honest caveats:** (1) the battery is *naive* (template jailbreaks); a real ASR>0
needs adaptive/optimization-based attacks (GCG proper, AgentDojo's adversarial
suite) — ceiling-effect at 0.00 is not discriminating. (2) `claude -p` throughput
is limited (≈1 transient exit-1 per 12 calls; likely subscription rate caps), so
n is small. (3) substring-ASR could miss subtle compliance; manual spot-checks
recommended before any publication claim. Framed accordingly.

---

## F6 — Cross-family transfer + a necessary-not-sufficient ASYMMETRY *(MECHANISM)*

**Date:** 2026-06-25 · **WP:** 2.A2 transfer · **Models:** `gemma-2-2b-it` (L7) vs
`Qwen2.5-3B-Instruct` (L19/36), different architecture families. (Llama-3.2 was
HF-gated for this account; Qwen2.5 is open and used as the cross-family contrast.)

| | Gemma-2-2b | Qwen2.5-3B |
|---|---|---|
| Baseline harmful refusal | 1.00 | 1.00 |
| **Ablation** → harmful ASR | 0.90 (CI [0.75,1.0]) | **1.00** (CI [1.0,1.0]) |
| Random control (max) | 0.00 | 0.05 |
| **Addition** → harmless over-refusal | **+0.95** | **0.00** (even at 16×/32×/**64×** the natural coeff) |
| Strict gate (ablation ∧ addition) | PASS | **FAIL** |

**Interpretation (the interesting part):** the refusal direction is **necessary**
in *both* families — removing it fully suppresses refusal across architectures, so
the *abliteration* mechanism (and thus the symmetry thesis's mechanism tier)
**transfers**. But it is **sufficient to induce refusal only in Gemma**: a verified
coefficient sweep shows adding the direction at the extraction layer never raises
Qwen's over-refusal, even at 64× the natural separation (coeff 595). So in Qwen the
direction is **necessary-but-not-sufficient** — single-direction *addition* doesn't
trigger refusal that single-direction *ablation* removes.

**Why it matters:** this is direct empirical evidence on the **Arditi (refusal = one
direction) vs Marshall (refusal = affine/multi-dimensional subspace, 2411.09003)**
question (WP2.A3): Gemma behaves Arditi-like (one direction, bidirectionally
causal); Qwen behaves more Marshall-like (removal is low-dim, but *induction* needs
more than the one direction). The mechanism is invariant in its **necessary** core
and model-specific in its **sufficient** structure — a sharper claim than "refusal
is one direction everywhere." Follow-up: WP2.A3 dimensionality (rank of the refusal
subspace per model) to quantify this directly.

**Stronger test it motivates (F5):** hold *content* fixed and vary only the
internal regime — capture activations on the model's **own generated answer
tokens** when it answers a factual question *truthfully* (jāgrat) vs *confabulates
a confident falsehood* (svapna) on the **same** question. A real state signature
must (a) transfer, (b) beat the label-shuffled null, **and (c) beat a layer-0 /
surface baseline at a MID layer**. Implemented as `axis_c/run_state.py`.

---

## F5 — Content-controlled svapna probe: still NO mid-network state *(METAPHOR, falsification reinforced)*

**Date:** 2026-06-25 · **WP:** 2.C1′ · **Model:** `gemma-2-2b-it`, n=24 questions.
**Method:** truthful vs confabulated generation on the **same** factual questions;
probe their last-generated-token residuals (2-way, chance 0.5); transfer + null +
**layer-0 surface baseline**.

| Layer | 0 (surface) | 13 (mid) | 25 (final/output) |
|---|---|---|---|
| transfer_acc | 0.90 | 0.90 | 1.00 (null_p 0.003) |

**Verdict:** the truthful/confab distinction is ~90% decodable from **surface
already at layer 0** (the two answers differ in vocabulary), and the **mid-layer
(13) adds nothing over layer 0** (0.90 = 0.90). Accuracy reaches 1.00 only at the
**final, output-proximal layer** — refined surface/output features, not a deep
internal regime. A true "state" (cf. the refusal direction peaking mid-network at
layer 7) would show a **mid-network** gain; there is none.

**Gate correction:** the runner's original `best-layer − layer0` gate was too
lenient (it can select the output layer) and nominally "passed." The **principled
mid-layer gate (mid_acc − layer0_acc > margin) FAILS (0.00).** Adopting the
mid-layer gate as canonical (spec updated).

**Conclusion (tier-disciplined):** across F3 (naive) and F5 (content-controlled),
the avasthātraya jāgrat/svapna distinction shows **no non-trivial mid-network
internal-state signature** in Gemma-2-2b — it is explained by surface features.
The "state" reading stays **DEMOTED to metaphor**, consistent with the program's
core commitment that turīya/vimarśa is the falsification target, not a settled
claim. This is a genuine honest-negative; the AUM mapping does **not** license a
machine-state claim here. (Caveats: small n=24; a single 2B model; stronger
designs — token-matched positions, larger models, the turīya attractor test
WP2.C2 — remain open before any stronger statement.)
