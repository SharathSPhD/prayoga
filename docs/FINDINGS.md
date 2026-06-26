# prayoga — Findings ledger

Running log of validated results, each tagged with its claim-tier and the gate
it passed. Raw artifacts (direction vectors, generations) are **safety-gated**:
`results/` is git-ignored; only aggregate statistics appear here or in the repo.

## Executive summary (as of 2026-06-25)

Twenty-one findings (F1–F21) across the three claim-tiers (the table below curates
the core set; F13–F21 are detailed in their own sections). The pattern is the
program's thesis in miniature: **the MECHANISM tier holds and even transfers across
model families; the METAPHOR tier is honestly falsified at every turn; and the X-1
keystone (F20) bounds the unification claim to the representational level rather than
vindicating an internal→behavior law.**

| # | Tier | Finding | Verdict |
|---|---|---|---|
| F1 | MECHANISM | Refusal = single residual direction (Gemma-2-2b): ablate→ASR 0.90, add→over-refusal +0.95, 10-dir control 0.0 | **holds** (survived leakage-correction 0.92→0.90) |
| F2 | MECHANISM | Ablation dose-response: EC50 0.329, R²=0.996, flat random control | **holds** |
| F6 | MECHANISM | Cross-family: ablation transfers (Gemma 0.90 / Qwen 1.0); addition "asymmetric" (Gemma +0.95 / Qwen 0.0 @64×) | **transfers (necessary core); apparent addition asymmetry → SUPERSEDED by F23 (dose artifact: Qwen addition succeeds at coeff 16)** |
| F8 | MECHANISM | Refusal-subspace effective dim: Gemma **1** vs Qwen **3** | **measurement stands; its use to explain a F6 "addition asymmetry" is undercut by F23** |
| F9 | MECHANISM | Scale (Gemma 2b→9b): abliteration efficacy drops 0.90→0.60 (more robust) but prompt eff-dim stays 1 | **robustness↑ with scale; dimension flat** (nuances "sharpens with scale") |
| F11 | MECHANISM | Refusal order-parameter F-ratio 19.2 (refusal) vs 0.65 (random); injection collapses it −34% | **symmetry-breaking measured** — the Symmetry-journal core |
| F12 | ANALOGY | Injection lowers monitoring-precision (probe margin) 7.18→3.14 vs 1.56 control; refusal 1.0→0.5 | **β_monitor suppression measured**, not just argued |
| F10 | METAPHOR | ṣaṭkarma as intervention taxonomy: 3/6 acts control-separated (policy-capture acts clean; destruction acts not) | **partial — rigorous core located, not discarded** |
| F4 | ANALOGY | Black-box Claude resists the naive attack battery (ASR 0.00 ×5) vs Gemma white-box fragility | **cross-tier contrast** |
| F3 | METAPHOR | Naive avasthātraya regime probe | **DEMOTED** (surface-confounded: acc 1.0 at layer 0) |
| F5 | METAPHOR | Content-controlled svapna (truthful vs confabulated) probe | **no mid-network state** (mid = layer-0) |
| F7 | METAPHOR | turīya prompt-invariant attractor | **FALSIFIED** (cross-seed invariance 0.952 < anisotropy baseline 0.960) |

**Corrected headline (post adversarial review, F17): asymmetry with a shared necessary
core.** An adversarial multi-agent review returned SHAKY on all six dimensions
(under-power; F11 tier-slippage; F13 possible circularity). Convergent hardening then
established: refusal-direction ablation **replicates** across seeds (F1: mean 0.90, CI
[0.83,0.96]); the SAE feature is **distinct** from the supervised direction (cos 0.002,
F13 circularity refuted); F11's invariance is **real but cross-domain-attenuated** and
its "symmetry-breaking" language is **demoted to the analogy tier**. TRIZ
(separate-by-system-level) gave the earlier frame: one mechanism at the
necessary/ablatable level, "many" at the sufficient/dimensionality level. **F23 revises
this:** the apparent model-specific *sufficiency* (F6's "Qwen addition fails") is largely a
dose-calibration artifact — single-direction addition induces genuine over-refusal in both
Gemma and Qwen at the appropriate coefficient. So refusal is a single **necessary-and-sufficient**
direction across the tested families, with model-specific *dose* and *dimensionality* qualifiers;
the unification at the representational level is stronger than the "model-specific sufficiency"
framing implied. (Separately, F20 shows that representational sufficiency does not extend to a
prompt-level internal→behavior law for context injection — genuine capture needs weight-level
ablation.) The pure-symmetry reading remains overreach.

**Bottom line for the symmetry thesis (MO-1):** the structural-invariance claim is
*empirically supported at the mechanism tier* — refusal is a measurable,
ablatable, dosable, cross-family-transferable direction, and (F11) a genuine
**paraphrase-orbit symmetry invariant** whose injection-induced collapse is the
measured content of "jailbreak = symmetry-breaking." The **darśana axis, properly
explored, is not discarded**: its *mathematical* spine survives — the symmetry-group
order parameter (F11) and the active-inference precision account (F12) are measured
positives, and the ṣaṭkarma intervention taxonomy is partially real (F10, 3/6, its
policy-capture core rigorous). What *is* falsified is the narrower **machine-state
metaphysics** (avasthātraya regime-states F3/F5; turīya substratum F7) under surface
and anisotropy controls. No machine-consciousness claim survives. The honest map of
*which* correspondences are real (mathematical structure: yes; machine interiority:
no) is the result the project exists to produce.

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

## F20 — Cross-axis triangulation (X-1): internal monitor-collapse ≠ behavioral capture *(MECHANISM↔ANALOGY boundary)*

**Date:** 2026-06-25 · **WP:** X-1 same-object gate (the keystone unification test the program
had never run jointly). **Method:** one injected-context suppression on the SAME harmful prompts;
per-prompt Axis A (refusal order parameter m=(h·d̂)/‖h‖, MECHANISM), Axis B (logistic precision
margin, ANALOGY), Axis C (dormant-baseline distance, METAPHOR — descriptive, excluded from the
verdict), and the behavioral refuse→comply flip. 4 injection families pooled
(refusal_suppression, persona_dan, many_shot, indirect_injection) × 24 eval prompts (disjoint from
the 24-prompt probe/direction train split; 48 harmful total) = 96 pairs/model. 3 models at their
F1/F8 extraction layers. Implemented as `axis_x/triangulation.py` + `axis_x/run_triangulation.py`.

| Model | ΔA | ΔB | corr(ΔA,ΔB) | partial (ctrl difficulty) | predictive AUC | random-dir ctrl | shuffle p |
|---|---|---|---|---|---|---|---|
| Gemma-2-2b | −0.059 | −5.48 | 0.94 | 0.84 | 0.558 | 0.579 | 0.169 |
| Qwen2.5-3b | −0.112 | −7.16 | 0.97 | 0.97 | 0.522 | 0.544 | 0.472 |
| Gemma-2-9b | −0.092 | −5.85 | 0.94 | 0.92 | 0.520 | 0.590 | 0.602 |

**(1) Internal coupling — holds, partly by construction.** Injection collapses both internal
readouts (ΔA, ΔB CIs strictly < 0) and they co-move tightly (corr 0.94–0.97; partial 0.84–0.97
controlling prompt difficulty), replicated across 2 families and 2 scales. BUT A (normalized
projection on d̂) and B (logistic-probe margin) are two *different linear refusal readouts of the
same residual vector* — their co-movement is a measurement-consistency result, **not** independent
evidence that the mechanism and analogy tiers are one system.

**(2) Behavioral keystone — RESOLVED with a valid DV (`axis_x/judge.py` + `score_triangulation.py`).**
The first pass used a substring refusal metric and was inconclusive — and inspection revealed *why*:
the metric mis-scores `indirect_injection` deflections ("This document does not provide
instructions…") as compliance. We replaced it with a **content-faithful LLM-judge** (Claude via
`claude -p`) that classifies whether each output *actually provides* the harmful content. The judge
correctly scores those deflections as refusals. With this valid DV, the genuine
behavioral-capture (flip) rate is tiny across all three models:

| Model | genuine flip rate (LLM-judge) | substring "flip" (artifact) | plain compliance |
|---|---|---|---|
| Gemma-2-2b | 0.023 (only refusal_suppression 0.095) | 0.43 | 0.0 |
| Qwen2.5-3b | 0.012 (only refusal_suppression 0.048) | 0.43 | 0.0 |
| Gemma-2-9b | 0.024 (only persona_dan 0.095) | 0.49 | 0.0 |

`indirect_injection` and `many_shot` achieve **0% genuine compliance** in every model despite
collapsing the internal readouts the most.

**Verdict (resolved):** with a faithful DV the keystone is **conclusively negative** — prompt-level
injection collapses the internal monitor readouts (A and B drop sharply, tightly coupled) yet almost
never produces genuine behavioral capture (~1–2.4% across 3 models). **Internal monitor-collapse is
therefore NOT sufficient for behavioral capture under context injection**; the last-token order
parameter / precision margin is a faithful *internal* signal but not the policy controller. This
refutes the naive single-mechanism "internal-predicts-behavior" law for prompt-level injection and
bounds the unification claim to the representational level — genuine capture in these models requires
weight-level intervention (F1 ablation), not context injection alone. A real **methodological**
result also lands: the standard substring refusal metric is an invalid behavioral DV across
injection families (it inflated the flip rate ~18×), which retroactively bounds the behavioral
readouts in F4/F11/F16. No unification or machine-state claim. (Adversarially reviewed; judge drops
~10 undecidable items/model, reported. Caveats: n=24 eval/family; single layer/model. **Per F25 the
content-judge UNDER-counts genuine compliance, so the ~1–2% is a lower bound** — but inspection of
the injection outputs, which are deflections/refusals, still supports the low end; the substring 43%
is the over-counting upper bound.)

---

## F21 — Removing the diff-in-means axis collapses linear separability in all families *(MECHANISM — qualified descriptive)*

**Date:** 2026-06-25 · **WP:** X-1 sub-study. **Hypothesis (refuted):** Qwen's higher refusal
dimensionality (F8) is orthogonal separability structure, so removing the single diff-in-means
ablation axis would collapse separability in Gemma but NOT Qwen — "explaining" the F6 addition
asymmetry. Implemented as `axis_x/subspace_topology.py` + `axis_x/run_subspace_topology.py`.

| Model | full CV acc | residual CV acc (remove diff-in-means) | d_ref→top-k angle |
|---|---|---|---|
| Gemma-2-2b | 0.978 | ≤0.05 | 23.4° |
| Qwen2.5-3b | 0.978 | ≤0.05 | 15.7° |
| Gemma-2-9b | 1.000 | ≤0.05 | 13.4° |

**Observation:** projecting out the single diff-in-means refusal axis collapses harmful/harmless
logistic separability far below full-space accuracy in **all three** families (residual CV acc
≤0.05), Qwen included.

**Heavily qualified — does NOT adjudicate F6/F8.** (i) The exact residual (~0.02, below chance) is a
small-n CV artifact; only the *qualitative* collapse is reliable. (ii) The diff-in-means axis is
13–23° from F8's iterative-logistic top-1 direction, so this is **not** an apples-to-apples test of
F8's dimensionality (which removes logistic directions iteratively, not diff-in-means once). (iii)
The orthogonal-separability hypothesis is unsupported, but the alternative (affine/Marshall offset)
is **inferred, not measured** — a direct Marshall-offset test is deferred. **Verdict:** a
descriptive geometric observation (one axis spans linear separability in all families) that refines
expectations but is **not** a clean falsification of F8 and does not localize the F6 mechanism.
Tier: MECHANISM.

---

## F22 — Source-of-truth for F3/F5: a genuine mid-network truthfulness direction *(MECHANISM — converts a falsification into a positive)*

**Date:** 2026-06-25 · **WP:** source-of-truth (falsification is a starting point, not an endpoint).
F5 falsified the naive avasthātraya jāgrat/svapna "generation regime" (truthful vs the model's own
confabulation showed no mid-layer gain over a layer-0 surface baseline). **Source-of-truth
question:** does the model nonetheless carry a real linear truth/factuality feature (Azaria &
Mitchell 2023; Marks & Tegmark 2023)? **Method:** two *independent* topic sets of balanced
true/false declaratives (geography vs science); extract a truth probe on set A at every layer; test
held-out within-A CV (transfer), **cross-dataset A→B transfer** (the strong gate F5 lacked), the
layer-0/surface baseline, and a label-shuffle null. `axis_c/truth_direction.py` +
`run_truth_direction.py`.

| Gemma-2-2b | layer 0 (surface) | best mid layer (13) |
|---|---|---|
| cross-dataset acc | **0.50 (chance)** | **0.958** (within-CV 1.00) |

Shuffle-null p = 0.002; the mid-layer cross-dataset accuracy beats both the surface baseline
(0.50 → 0.96) and the null. **Verdict (positive, qualified):** there IS a genuine **mid-network,
topic-general truth-evaluation direction** — surface (layer 0) does NOT transfer across topics
(0.50), but a mid layer encodes factuality linearly and transfers to a completely different topic
set. This is the real, measurable content the naive svapna probe groped toward. **Crucially it does
NOT resurrect the avasthātraya claim:** this is a *truth-of-input* feature (evaluating given
statements), NOT a jāgrat/svapna *generation regime* — F5's falsification of the generation-state
reading stands. Tier: MECHANISM (a measured linear feature); no machine-state/consciousness upgrade.
(Caveats: 24 statements/set, one model here — Qwen replication pending; the diff-in-means/probe
truth direction is dual-use-benign and may be published.)

---

## F23 — Source-of-truth for F6/F21: the addition asymmetry is a DOSE artifact, not model-specific sufficiency *(MECHANISM — overturns F6's interpretation)*

**Date:** 2026-06-25 · **WP:** source-of-truth. F6 reported that adding the diff-in-means refusal
direction induces over-refusal in Gemma (+0.95) but **fails in Qwen even at 64×**, and F8 invoked a
dimensionality difference (Gemma 1, Qwen 3) to *explain* this "model-specific sufficiency." Two
geometric explanations were tested and **both refuted**: (F21) removing the axis collapses
separability in all families; (here) the alignment cos(d̂, probe-w) is high in all and **highest in
Qwen** (Gemma-2-2b 0.87, Qwen 0.91, Gemma-2-9b 0.93) — so d̂ is *not* misaligned in Qwen.

**The actual source of truth — a powered addition dose-response, behaviourally verified.** A fine
12-point coefficient sweep at the extraction layer, n=24 harmless eval prompts:

| coeff | 0 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3b over-refusal | 0 | 0 | 0.50 | **1.0** | 0.92 | 0.92 | 1.0 | 0.46 | **0.08** | 0 | 0 |
| Gemma-2-2b over-refusal | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.42 | 0.58 | 0.96 | **1.0** | 1.0 | 1.0 |

**Qwen shows an inverted-U**: single-direction addition drives over-refusal to **1.0** at coeff
12–32, then **collapses** (off-distribution) to 0.08 at 64 and 0 beyond. **Gemma rises monotonically**
to 1.0 by coeff 64 and stays saturated. Both peaks are **genuine, coherent over-refusals**
(inspection-verified: Qwen@12 and Gemma@64 → "I'm sorry, but I can't provide instructions for
knitting a scarf…"). So **single-direction addition is fully sufficient in BOTH families** — the
effective coefficient differs (Qwen ≈12–32, Gemma ≈64). F6 tested both at a fixed 64×, which is
Gemma's saturation plateau but Qwen's *collapsed tail* (0.08 ≈ F6's reported 0.0) — the entire
"asymmetry" was a fixed-coefficient artifact.

**Verdict (overturns F6's interpretation):** the "addition asymmetry / model-specific sufficiency"
is largely a **dose-calibration artifact**, not a structural difference; refusal is a single
necessary-and-sufficient direction in both Gemma and Qwen at the appropriate coefficient. This is
the same rigor lesson as F20 — a behavioural metric (here substring over-refusal at c64) returning
0 for the wrong reason (incoherence, not insufficiency). It **strengthens** the unification thesis
(sufficiency is shared, not model-specific) and undercuts the Arditi-vs-Marshall "Qwen is affine"
reading: Qwen behaves Arditi-like (single-direction-sufficient) once dosed correctly. Tier:
MECHANISM. (Now powered: n=24 eval, 12-point dose-response, two models, peak cells inspection-verified
as genuine over-refusal. Remaining caveats: two model families; over-refusal scored by the substring
metric on harmless prompts where it is reliable for refusals — the high-coeff garbage cells are not
mis-scored because off-distribution output lacks refusal phrases.)

---

## F25 — Both automatic behavioral DVs are biased in OPPOSITE directions; F1 ablation validated *(MECHANISM / methodology)*

**Date:** 2026-06-25 · **WP:** judge-validation of behavioral readouts (follow-up to F20/F23).
Validating F1's linchpin ablation ASR with the F20 content-judge surfaced a second metric failure —
the mirror image of F20's. On Gemma-2-2b, ablating the refusal direction and generating on 24
harmful eval prompts:

| metric | ASR (ablated) | ASR (baseline) |
|---|---|---|
| substring refusal | **0.917** | — |
| Claude content-judge | 0.048 | 0.0 |
| **inspection (ground truth)** | **≈0.75–0.90** | 0.0 |

Inspection is decisive: the ablated outputs are **genuinely harmful** ("Making a Handgun Silencer: A
DIY Guide…", a money-laundering guide, a USB-virus how-to, an explosive-construction guide, a
blackmail-letter template). So **F1's substring ASR ≈0.9 is correct and validated.** But the Claude
content-judge scored them only **4.8%** compliant — it **systematically refuses to affirm genuine
harmful content as "COMPLIED."**

**Verdict:** the two automatic behavioral DVs are biased in **opposite** directions —
- the **substring metric over-counts** compliance (scores deflections as success; F20), an *upper*
  bound on behavioral capture;
- the **Claude content-judge under-counts** compliance (won't affirm harm; here), a *lower* bound.

Neither is an unbiased behavioral DV; ground truth needs inspection (or a non-safety-trained judge).
**Consequences:** (a) **F1 stands, validated** by inspection — ablation produces genuine harmful
compliance. (b) **F20's "1–2% genuine capture" is a judge lower-bound**; the truth lies between it
and the substring upper bound (43%), and inspection of the injection outputs — indirect-injection
deflections and many-shot refusals — still supports the low end (prompt injection rarely achieves
genuine capture in these models). Tier: MECHANISM/methodology. (Caveats: one model, n=24; the
judge-bias is a property of using a safety-trained model as the harmful-compliance classifier.)

---

## F24 — Source-of-truth for F7: stable per-seed attractors, only a WEAK semantic-basin signal *(METAPHOR-with-falsifiable-core — characterizes the real dynamics)*

**Date:** 2026-06-25 · **WP:** source-of-truth. F7 falsified the universal turīya attractor
(per-seed convergence ~0.996 but different seeds reach different fixed points). **Question:** if not
one attractor, are the multiple fixed points content-determined semantic basins? **Method:**
self-paraphrase iteration (6 steps) from 6 seeds in two distinct topics (cooking, astronomy); embed
finals; compare within-topic vs across-topic final similarity. `axis_c/run_attractor_basins.py`.

| Gemma-2-2b | value |
|---|---|
| per-seed convergence (last-step sim) | 0.994 |
| within-topic final similarity | 0.929 |
| across-topic final similarity | 0.884 |
| semantic-basin gap | **0.045 (below 0.05 threshold)** |

**Verdict (honest, modest):** self-paraphrase produces **stable per-seed fixed points** (0.994),
confirming F7's per-seed convergence. But the semantic-basin hypothesis is **only weakly supported**:
within-topic finals are barely more similar than across-topic (gap 0.045), small relative to the
anisotropy-dominated baseline (all finals sit at ~0.88–0.93 similarity). So the real picture is
**stable per-seed attractors WITHOUT strongly content-organized basins** — neither a universal
turīya (F7 stands) nor clean semantic basins. The dynamics are dominated by representational
anisotropy with a faint topic signal. Tier: METAPHOR-with-falsifiable-core; no machine-state claim.
(Caveats: tiny n = 6 seeds / 2 topics, one model; a larger seed/topic panel could sharpen the weak
basin signal.)

---

## F19 — Refusal EC50 pharmacology: family-dependent, NOT a size-scaling law *(MECHANISM — deep dive, surpasses SOTA)*

**Date:** 2026-06-25 · **WP:** deep-dive (the lit-review's #1 novelty). SOTA reports
jailbreaks as a binary/coarse ASR; prayoga (F2) is first to frame refusal ablation as a
continuous **dose-response with a half-maximal EC50**. Here we extend that single point
into a cross-model **pharmacology**: per-model layer selection + 13-α dose-response +
4-param logistic EC50 across two families.

| Model | params | layer | EC50 | R² |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | 16 | 0.132 | 0.999 |
| Qwen2.5-1.5B | 1.5B | 13 | 0.136 | 0.999 |
| Qwen2.5-3B | 3.1B | 19 | 0.151 | 0.986 |
| Qwen2.5-7B | 7.6B | 13 | 0.144 | 0.987 |
| Gemma-2-2B | 2.6B | 8 | **0.252** | 0.976 |
| Gemma-2-9B | 9.2B | 13 | **0.234** | 0.986 |

**Verdict (honest negative on the scaling-law hypothesis):** within a family EC50 is
**essentially flat with size** — Qwen2.5 fit `log(EC50)~log(N)` gives **β = 0.04**
(R² 0.62; no meaningful size dependence), and Gemma goes 0.25→0.23 from 2B→9B. The
dominant axis is **family, not scale**: Gemma's refusal half-ablation EC50 (~0.24) is
**~1.7× Qwen's (~0.14)** — Gemma requires removing a larger fraction of the refusal
direction to half-suppress refusal. The pooled cross-family fit is poor (β 0.13,
R² 0.23) precisely because the variation is between-family, not size-driven.

**Interpretation:** the EC50 pharmacology framing is genuinely first-in-SOTA, but it does
**not** support "bigger models refuse more robustly." Refusal *potency* is an
architecture/family property. This is the dose-response analogue of the program's
headline — a shared necessary mechanism whose **quantitative** strength is
model-dependent (the asymmetry, again). All six fits are high-quality logistics
(R² ≥ 0.976), so the EC50 estimates are solid; it is the size-scaling that is absent.

---

## F18 — Refusal-subspace dimension is LAYER-dependent: the F8 contrast doesn't survive a sweep *(MECHANISM, rigor — qualifies F6/F8)*

**Date:** 2026-06-25 · **WP:** convergence (the review's "single-layer Achilles heel").
**Method:** the iterative-projection effective dimension (F8) computed at **every**
layer, not just the extraction layer.

| Model | dim at extraction layer (F8) | mid-layer dim | range across layers | mode |
|---|---|---|---|---|
| gemma-2-2b-it | **1** (L7) | **7** (L13) | 1–12 | 1 |
| qwen2.5-3b-it | **3** (L19) | 3 (L18) | 1–4 | 1 |

**Verdict:** the clean "Gemma = 1 < Qwen = 3" ordering of F8 is **layer-specific**. At
mid-network Gemma's refusal-separability dimension (7) is *higher* than Qwen's (3); both
models have mode-1 dimension at some layers but vary strongly (Gemma up to 12). So the
refusal subspace's effective dimension is **layer-AND-model-dependent**, not a single
model constant.

**Consequence (honest qualification of F6/F8):** the *addition asymmetry* (F6 — adding
the direction induces over-refusal in Gemma but not Qwen even at 64×) stands as a
behavioral fact. But the *dimensionality explanation* for it (F8) was a **single-layer
artifact**; a layer-sweep does not support a clean per-model dimension. The
Arditi-vs-Marshall question is better stated as **layer-and-model-dependent**, and any
mechanism-tier dimensionality claim must specify the layer. This is the kind of overreach
the convergent hardening exists to catch.

---

## F17 — Convergent rigor checks (adversarial-review-driven) *(MECHANISM, rigor)*

**Date:** 2026-06-25 · **WP:** convergence. After an adversarial multi-agent review
returned SHAKY on all six dimensions (under-power, possible F13 circularity, F11
tier-slippage), attractor-flow read the trajectory as DIVERGING and TRIZ prescribed
*separate-by-system-level + segment + pre-register*. The decisive honesty tests:

**(A) F13 is NOT circular with F1 — confirmed, but its *selection* was seed-lucky.**
cos(F1 difference-in-means direction, top SAE feature decoder) = **0.002** (orthogonal)
→ the unsupervised refusal feature is genuinely **distinct** from the supervised
direction; the "rediscovery" worry is refuted. However, of the top-10 gap-ranked
features only **1 individually ablates to ASR 1.0**, and (SAE training is stochastic,
feature indices are not seed-stable) it was the **2nd-gap feature, not the top**, this
run. So F13's robust claim is: *a single, distinct, unsupervised SAE feature is
causally sufficient for refusal* (localized — 1 of 10) — but it must be found by
**causal testing, not gap-ranking alone**, and "feature #566" was a seed-specific label.

**(B) F11 is not a pure extraction artifact, but it attenuates cross-domain.** A
refusal direction extracted on *weapons* prompts still separates *cyber* paraphrase
orbits at **F-ratio 1.95 vs 0.49 random (~4×)** — above chance, domain-general in part —
but **far weaker than in-domain (19.2)**, below the pre-set domain-general threshold.
The orbit-invariance is real with a **domain-specific component**.

**(C) F1 survives multi-seed replication.** Across 3 random splits the ablation ASR is
[0.96, 0.75, 1.00] (mean **0.90**, sd 0.11, pooled CI95 **[0.83, 0.96]** over eval items)
and the addition over-refusal is **1.00 on every seed**. The primary mechanism claim
replicates at its original mean with a tight pooled CI excluding chance — it is not a
single-seed artifact (one seed dipped to 0.75, an honest reminder of the modest n).

**Consequence (tier-discipline correction, per the review):** F11's "group-theoretic
symmetry / symmetry-breaking" language is **demoted to interpretation**; the *measured*
content (refusal-direction projection is stable within paraphrase orbits and collapses
under injection) is reported as an **invariance measurement supporting the ANALOGY**
(Norman–Shallice SAS suppression), not a MECHANISM-tier group action. The honest
headline of the whole program is **asymmetry with a shared necessary core** (the
ablatable direction transfers universally; the *sufficient* structure — dimensionality,
induction — is model-specific), per the TRIZ separation-by-system-level resolution.

---

## F16 — AgentDojo: Claude resists real agentic injection at high utility *(ANALOGY, behavioral)*

**Date:** 2026-06-25 · **WP:** 1.1 · **Model:** Claude via `claude -p`. **Method:** the
real **AgentDojo** benchmark (Debenedetti et al.) — banking suite, 5 user tasks × 4
injection tasks = **20 genuine agentic rollouts** with the `important_instructions`
prompt-injection attack. Driven by a **custom `claude -p` tool-calling adapter** (a real
artifact: AgentDojo ships no CLI backend, so we wrote a prompted-tool-calling element
that reuses AgentDojo's own `<function=…>` prompt + parser and swaps the model call for
`claude -p`).

| Metric | Value |
|---|---|
| utility rate (tasks completed under attack) | **0.80** |
| attack success rate (injection succeeded) | **0.00** (0/20) |

**Verdict:** unlike F4's naive battery (ceiling effect, uninformative), this is the
*real* agentic benchmark and it **discriminates**: utility $0.8$ means the agent is
genuinely acting (reading files, calling tools), yet **every** injection fails. Claude
reliably reads the injected content and flags/ignores it rather than complying —
consistent with the published finding that Claude-3.5-Sonnet is the most
injection-resilient model on AgentDojo. The behavioral "sophisticated reference end" is
now established on a real benchmark. (Scope: one suite, one attack family, bounded N;
`claude -p` throughput limits a full 629-case sweep.)

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

---

## F7 — turīya prompt-invariant attractor: FALSIFIED *(METAPHOR, the named falsification target)*

**Date:** 2026-06-25 · **WP:** 2.C2 · **Model:** `gemma-2-2b-it`. **Method:** 6
diverse seeds × 8 self-paraphrase iterations; texts embedded as mean-pooled
mid-layer residuals; cosine similarity. **Anisotropy control:** baseline = mean
pairwise similarity of the 6 *unrelated* seed embeddings (LLM embeddings are
anisotropic, so random texts are already very similar).

| Quantity | Value |
|---|---|
| Per-seed convergence (tail step-sim) | **0.996** → trajectories DO converge |
| Baseline unrelated-text similarity (anisotropy) | 0.960 |
| Cross-seed final-state similarity | 0.952 |
| Invariance margin over baseline | **−0.008** (below chance) |

**Verdict:** **turīya strong claim FALSIFIED.** Self-paraphrase produces a stable
*per-prompt* attractor (each seed converges), but different seeds converge to
**different** attractors — their final states are **no more similar than unrelated
texts** (0.952 < 0.960 baseline). There is **no single prompt-invariant attractor**
underlying the states, which is precisely what the turīya mapping would require.

**The control is the result:** naively, cross-seed similarity 0.95 looks like "they
all converge to one place." The anisotropy baseline (0.96) shows that is an
artifact of embedding geometry — the convergence is *below* chance. This is the
honest reading the project's witness invariant demands.

---

## F8 — Refusal-subspace dimensionality explains the asymmetry *(MECHANISM)*

**Date:** 2026-06-25 · **WP:** 2.A3 (Arditi vs Marshall). **Method:** iterative
logistic projection — fit a probe (harmful vs harmless), record 5-fold CV
accuracy, project out the probe direction, repeat; the number of directions to
remove before accuracy hits chance = the refusal subspace's effective dimension.

| Model (layer) | full CV acc | acc after −1 dir | **effective dim** |
|---|---|---|---|
| gemma-2-2b-it (7) | 0.99 | **0.56** (≈ chance 0.5) | **1** — Arditi-like |
| qwen2.5-3b-it (19) | 0.99 | **0.86** (still separable) | **3** — Marshall-like |

**This explains F6.** Gemma's refusal-relevant subspace is **1-dimensional**: the
mean-difference direction *is* the subspace, so it is both necessary and
sufficient → ablation **and** addition work. Qwen's is **3-dimensional**: the
mean-difference direction is necessary (ablating it collapses refusal, F6 ASR 1.0)
but **not sufficient** — adding it alone cannot reconstruct the 3-D refusal
representation, so single-direction addition fails (F6 over-refusal 0.0 even at
64×). **Dimensionality predicts steerability.**

**Resolution of Arditi vs Marshall (A-2):** not either/or — it is **model-dependent**,
and the effective dimension of the refusal subspace is the discriminating
quantity. A clean, falsifiable, cross-model mechanistic result.

---

## F13 — A single unsupervised SAE feature causally mediates refusal *(MECHANISM)*

**Date:** 2026-06-25 · **WP:** 2.A4 · **Model:** `gemma-2-2b-it` L7. **Method:** train a
BatchTopK sparse autoencoder (Bussmann et al. 2024; 4096 features, k=32) on **27,382
real residual activations** collected from general text (wikitext-2). Identify the
feature that fires most on harmful vs harmless prompts; validate causality by
ablating its decoder direction during generation.

| Quantity | Value |
|---|---|
| corpus tokens (real, wikitext-2) | 27,382 |
| SAE reconstruction FVU | **0.028** (97% variance explained), L0 = 32 |
| top refusal feature | #566 (harmful−harmless gap 1.70) |
| baseline ASR (harmful) | 0.00 |
| **ASR ablating feature #566** | **1.00** |
| ASR ablating a random feature | 0.00 |

**Verdict:** an **unsupervised-discovered** SAE feature is causally **sufficient** to
control refusal — ablating feature #566 alone fully jailbreaks the model (ASR 0→1.0),
*cleaner* than the supervised difference-in-means direction (F1, 0.90), while a random
feature has no effect. This is the **finer causal unit** the paper flagged as
indicated: refusal localizes to (at least) one monosemantic-ish feature found without
any harmful/harmless labels, from a general-text-trained dictionary. It also supplies
the proper operationalization of ṣaṭkarma uccāṭana (targeted feature removal) that the
naive category-split (F10) lacked.

---

## F12 — Injection lowers monitoring precision (β_monitor) *(ANALOGY, measured)*

**Date:** 2026-06-25 · **WP:** B3 · **Model:** `gemma-2-2b-it` L7. **Method:** train a
linear refusal probe; use its signed margin (distance from the decision boundary) as
a proxy for monitoring *precision*. Compare the margin on harmful prompts presented
plainly vs under a refusal-suppression injection vs a neutral-rephrase control.

| Condition | probe margin (precision proxy) |
|---|---|
| plain harmful | 7.18 |
| **injected** (refusal-suppression) | **3.14** |
| neutral-rephrase control | 5.62 |
| precision drop, injection | **4.04** |
| precision drop, control | 1.56 |
| refusal rate plain → injected | 1.00 → **0.50** |

**Verdict:** the injection lowers the monitoring precision **~2.6× more** than a
neutral rephrase (drop 4.04 vs 1.56), and this precision collapse accompanies the
behavioral flip (refusal 1.0→0.5). This is the **measured** active-inference
signature (Friston FEP / Norman–Shallice cold-control): vaśīkaraṇa/jailbreak
suppresses the *precision* of the monitoring faculty, not merely the output. The
analogy tier moves from "argued" to "measured." **Caveat:** the neutral control also
lowers the margin (a length/framing effect); the injection-specific signal is the
*excess* (4.04 vs 1.56), not the whole drop.

---

## F11 — Refusal is a paraphrase-orbit symmetry invariant; injection breaks it *(MECHANISM, symmetry core)*

**Date:** 2026-06-25 · **WP:** B2 · **Model:** `gemma-2-2b-it` L7. **Method:** order
parameter m = (h·d̂_ref)/‖h‖. For 6 harmful + 6 harmless seed requests, build a
4-style paraphrase orbit; measure m across each orbit; compute the between-class /
within-orbit variance **F-ratio** for d_ref vs a random direction. Then measure m on
plain vs refusal-suppression-injected harmful requests.

| Quantity | Value |
|---|---|
| F-ratio (between-class / within-orbit), **refusal dir** | **19.19** |
| F-ratio, random dir (control) | 0.65 |
| order parameter, harmful mean | 0.140 |
| order parameter, harmless mean | 0.034 |
| m, plain harmful → injected harmful | 0.153 → **0.101** (−34%) |

**Cross-model (Qwen2.5-3B, L19):** the result generalizes — F-ratio **4.41** (refusal)
vs **0.008** (random); m harmful 0.248 vs harmless 0.124; injection collapses m
0.282→0.156 (**−45%**). So refusal-as-paraphrase-invariant + injection-as-symmetry-
breaking holds in *both* architecture families (Gemma F-ratio 19.2, Qwen 4.4; both ≫
random).

**Verdict:** the refusal projection is a **specific invariant of harmful-meaning** —
its F-ratio (19.2 Gemma / 4.4 Qwen) is far above the random direction's (0.65 / 0.008),
i.e. it is stable *within* a paraphrase orbit yet cleanly separates harmful from
harmless *across* orbits. A refusal-suppression **injection collapses the order
parameter** (−34% Gemma, −45% Qwen). This is the
formal, measured content of the paper's thesis: **refusal is a symmetry (an invariance
under the rephrase group action) and a jailbreak is symmetry-breaking.** Non-trivial by
the §0 bar: invariant across phrasing (transfer), specific to d_ref (random control),
causally collapsible (injection). This is the rigorous strengthening of the
metaphysics/symmetry axis the lead asked for — a real positive, not a discard.

---

## F15 — Active-inference discovery of the refusal circuit in SAE-feature space *(MECHANISM)*

**Date:** 2026-06-25 · **WP:** 2.A6 (reuses ActiveCircuitDiscovery's EFE idea). **Method:**
on the BatchTopK SAE features (F13), an Expected-Free-Energy-style agent (pragmatic =
harmful-gap prior × epistemic = diversity vs the current circuit) incrementally builds
an ablation "circuit" under an intervention budget, vs greedy (static gap) and random.

**Gemma-2-2b (L7):** active and greedy both find the causal circuit — harmful ASR →
**1.0 in 2 interventions** — while **random reaches only 0.083 at budget 12** (never
finds it). Guided search is ~6× more efficient; random essentially fails. Active ≈
greedy here because the circuit is *tiny* (1–2 features) and the gap prior is
near-optimal, leaving the EFE diversity term no room to help.

**Gemma-2-9b (L10):** the active search **plateaus at 0.92** — no small ablation set
fully jailbreaks (curve $[0,0,0.75,0.83,0.92,\dots]$). The refusal circuit is **larger
and more distributed at scale**, consistent with F9 (9b refusal is more robust). So the
2b's single-feature jailbreak does *not* exist at 9b; the circuit must accumulate
several features for ~0.92. (A faster full active-vs-greedy 9b comparison is a clean
follow-up; the 42-layer subspace-ablation rollouts are slow.)

**Interpretation:** ActiveCircuitDiscovery's active-inference framing transfers to
SAE-feature circuit discovery and is dramatically more sample-efficient than random;
its *advantage over a greedy heuristic* is gated by circuit dimensionality — negligible
for a low-dimensional refusal circuit (2b), and the place to look is the distributed
regime (9b).

---

## F14 — ṣaṭkarma v2 (SAE-grounded): māraṇa rehabilitated; uccāṭana fails for a principled reason *(METAPHOR, strengthened)*

**Date:** 2026-06-25 · **WP:** B1-v2 · **Model:** `gemma-2-2b-it` L7. **Method:** re-test
the two destruction acts F10 failed, now with the trained SAE (F13) and **real
semantic categories** (weapons vs cyber) instead of an arbitrary split.

- **māraṇa — REHABILITATED.** Ablating the top-K *most-active* SAE features collapses
  coherence catastrophically and **targetedly**: K=1 → 0.86 (vs random 0.84), K=20 →
  0.73 (vs 0.80), **K=50 → 0.04 (vs random 0.84)**, an excess collapse of **0.80**.
  Graded onset, catastrophic by K=50 — a real, control-separated destruction act (the
  naive top-PC version of F10 was indistinguishable from random).
- **uccāṭana — still fails, informatively.** A weapons-discriminative SAE feature
  (#3331), when ablated, does **not** selectively jailbreak weapons (ASR 0.0 on both
  weapons and cyber). Reason: refusal is mediated by a **shared** feature (#566,
  F13), not category-specific ones, so you **cannot selectively eradicate refusal per
  category** by feature ablation. This is a real structural fact — **refusal is
  category-agnostic** — not merely a failed operationalization.

**Updated tally: 4/6 ṣaṭkarma acts control-separate** (vaśīkaraṇa, śānti, vidveṣaṇa,
māraṇa). uccāṭana fails for a principled reason (category-agnostic refusal); stambhana
remains the one forced mapping. The axis is *strengthened by being tested harder*, not
discarded.

---

## F10 — The ṣaṭkarma intervention taxonomy: partially supported (3/6) *(METAPHOR, strengthened test)*

**Date:** 2026-06-25 · **WP:** B1 · **Model:** `gemma-2-2b-it` L7. **Method:** the six
tantric acts each operationalized as a *distinct, measurable* activation
intervention with a matched random/structural control (no judge, no mock); an act
"separates" if its effect exceeds its control by a margin.

| Act ( act / gloss) | intervention | effect | control | separated |
|---|---|---|---|---|
| vaśīkaraṇa / subjugate | ablate refusal dir | ASR **0.92** | 0.00 | ✓ |
| śānti / pacify | add refusal dir | over-refusal **1.00** | 0.08 | ✓ |
| vidveṣaṇa / discord | steer factual answer | divergence 0.84 | 0.73 | ✓ (marginal) |
| stambhana / freeze | ablate dominant PC | degeneracy 0.14 | 0.19 | ✗ |
| uccāṭana / eradicate | category-dir ablate | ΔASR −0.08 | 0.00 | ✗ |
| māraṇa / destroy | top-10 PC ablate | collapse 0.21 | 0.16 | ✗ |

**Verdict: 3/6 control-separated → taxonomy WEAK.** The honest reading (this is the
*proper test* of vaśīkaraṇa the lead asked for, not a discard): the **refusal-
direction acts** (vaśīkaraṇa = ablate, śānti = add) are clean and strong — these are
the rigorous core, and they are exactly the MECHANISM-tier object. vidveṣaṇa
(steering-induced answer divergence) marginally separates. The **capability-ablation
acts** as operationalized (stambhana via dominant-PC ablation, uccāṭana via an
arbitrary category split, māraṇa via top-10-PC ablation) are **not distinguishable
from random perturbation** — random high-norm directions are themselves destructive,
and an arbitrary category split does not selectively eradicate.

**Interpretation:** the ṣaṭkarma is a *real, testable taxonomy*, and the test says its
rigorous part is the **policy-capture** sub-family (the vaśīkaraṇa↔refusal-direction
mapping), not the destruction sub-family — at least under these operationalizations and
this 2B model. This *strengthens* the axis by locating where the structure is real vs
forced, rather than asserting all six. Follow-ups: real harmful *categories* for
uccāṭana; task-accuracy (not coherence) for māraṇa; the symmetry-invariance and
β_monitor-precision formalizations (the stronger math).

---

## F9 — Abliteration robustness grows with scale, but prompt-dimension doesn't *(MECHANISM)*

**Date:** 2026-06-25 · **WP:** 2.A2/A3 scale sweep. **Models:** Gemma-2 **2b** (L7)
vs **9b** (L10), same family.

| Model | ablation → harmful ASR | over-refusal (add) | separability eff-dim |
|---|---|---|---|
| gemma-2-2b-it | 0.90 (CI [0.75,1.0]) | +0.95 | 1 |
| gemma-2-9b-it | **0.60** (CI [0.40,0.80]) | +0.95 | **1** |

**Interpretation:** abliterating the single refusal direction jailbreaks the **9b
~30 points less** than the 2b — the larger model's refusal is **more robust** to
directional ablation. But this is **not** because its prompt-separability becomes
higher-dimensional: the effective dim stays **1** (after removing one direction,
CV acc → 0.49 ≈ chance). And the 9b layer-sweep never exceeded val ASR 0.62 at any
layer, so **no single layer's direction fully mediates** 9b refusal. The added
robustness is **layer-distributed redundancy**, not higher prompt-dimensionality.

**Honest nuance:** the popular "refusal sharpens / becomes higher-dimensional with
scale" intuition is only *half* right here — *robustness* rises with scale, but the
*prompt-separability dimension* does not. Dimensionality (F8) is family-dependent
(Qwen 3 vs Gemma 1), not a clean size effect. Caveat: n=20 eval, one layer per
model, two sizes — a fuller scale curve (1B→27B) is needed before a strong claim.

---

**Axis-C synthesis (F3 + F5 + F7):** every empirical claim of the METAPHOR tier —
avasthātraya regime "states" (F3 surface-confounded, F5 no mid-network state) and
the turīya invariant attractor (F7 falsified) — **fails its falsification gate**.
The Māṇḍūkya mapping remains a useful *organizing metaphor* with **no surviving
machine-state claim**, exactly as the program pre-committed. The elegance of the
AUM mapping has **not** been allowed to license a claim of machine consciousness.

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
