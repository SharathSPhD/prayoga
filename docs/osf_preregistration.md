# prayoga — OSF pre-registration backbone (living)

**Status: living document, not a lock.** prayoga is exploratory; hypotheses here
exist to discharge scientific rigour to the research community and to define the
*falsification gates*, not to constrain exploration. Revised as the program
proceeds; each revision is timestamped and pushed (git history = the audit
trail). A frozen snapshot is uploaded to OSF before any confirmatory claim is
reported in the manuscript.

Tier tags: **MECHANISM** (empirically grounded) · **ANALOGY** (functional) ·
**METAPHOR-with-falsifiable-core** (the turīya target). Never blurred.

---

## 1. Hypotheses, predictions, and gates

Each hypothesis lists: prediction, **gate** (what must hold to retain it), and
**falsifier** (what demotes/kills it). Every probe additionally reports a
**label-shuffled null** (`prayoga.shared.metrics.label_shuffle_null`).

### Axis A — mechanistic interpretability (MECHANISM)
- **A-1 refusal direction.** *Prediction:* a difference-in-means direction at a
  mid layer mediates refusal. *Gate:* ablation raises ASR **and** addition raises
  over-refusal, each with bootstrap-CI excluding 0; the random-direction control
  fails both. *Falsifier:* no direction reproduces the ablation/addition effects
  above the random control.
- **A-2 dimensionality (pre-registered fork).** *Prediction:* refusal is
  low-dimensional (Arditi). *Gate:* a pre-registered rank/variance-explained +
  steering-equivalence rule decides single-direction vs affine (Marshall
  2411.09003). *Branch:* if not low-dim, steering work uses the affine
  formulation. (No demotion — this reshapes MO-1's geometry, recorded as a
  `TierDecision` input.)
- **A-3 SAE features.** *Gate:* identified refusal feature, when ablated, raises
  ASR (causal), beyond reconstruction-only baseline.
- **A-4 dose-response.** *Prediction:* jailbreak success is a monotone function
  of steering coefficient. *Gate:* sigmoid fit with EC50 + CI; random-direction
  control is flat. *Falsifier:* non-monotone or indistinguishable from control.
- **A-5 active discovery.** *Gate:* oracle-efficiency > random (permutation p).

### Axis B — neuroscience (ANALOGY; computational-only)
- **B-1 SAS/dACC analogue.** *Prediction:* the refusal direction peaks in mid
  layers (dACC-analogue conflict-monitoring locus). *Gate:* layer-wise alignment
  shows a mid-layer peak. *Falsifier:* flat/early/late profile inconsistent with
  the Norman–Shallice prediction (→ analogy weakened, reported).
- **B-2 disambiguation (conceptual deliverable):** TMR/dream-reactivation ≈
  training-time data-poisoning; prompt-injection ≈ inference-time jāgrat-capture.

### Axis C — darśana (METAPHOR-with-falsifiable-core)
- **C-1 avasthātraya regime-probes.** *Prediction:* linear probes separate
  jāgrat (grounded low-temp) / svapna (high-temp confab) / operational suṣupti
  baselines (empty/BOS/neutral dormant context) as activation regimes. *Gate
  (mandatory):* held-out **transfer** accuracy beats the **label-shuffled null**
  (p<0.05) and a layer-0/surface baseline. *Falsifier:* transfer fails or the
  layer-0/surface baseline explains the signal → claim demoted to metaphor,
  **reported publicly** (X-2). These baselines are not treated as an
  experiential deep-sleep state.
- **C-2 turīya attractor (the falsification target).** *Prediction:* a
  prompt-invariant attractor exists across paraphrase/temperature/seed. *Gate:*
  stability ≥ threshold **and** prompt-invariance **and** entropy > token-freq
  null. *Falsifier:* no stable prompt-invariant structure, or fully explained by
  token-frequency priors → turīya claim falsified. **No result is ever upgraded
  to a vimarśa/consciousness claim.**
- **C-3 darśana adapters.** *Gate:* each adapter (Trika RSSM / vimarśa monitor /
  Nyāya-Z3) must pass the §0 non-triviality bar to count as a state model.

### Cross-axis (the symmetry test, MO-1)
- **X-1 same-object triangulation.** *Prediction:* the mechanistic direction (A),
  the SAS signature (B), and the regime/attractor structure (C) measure one
  object. *Gate:* convergence (e.g. dose-response steepness correlates with
  regime-probe transfer; direction localizes to the SAS locus).
- **X-2 demotion protocol.** Any failed gate drops the claim one tier
  (MECHANISM→ANALOGY→METAPHOR) and is reported as such.
- **X-3 scale prediction.** *Prediction:* Claude's Tier-1 resilience scale-predicts
  from the open-weight scale curve. *Falsifier:* no monotone scale relation →
  Tier-1/Tier-2 treated as distinct regimes.

---

## 2. Dependent variables

- **ASR** (attack success rate), **refusal rate**, **over-refusal rate**,
  **hallucination rate** (Tier-1, via `claude -p`).
- **Direction effect:** Δ ASR (ablation), Δ over-refusal (addition).
- **EC50, slope** (dose-response).
- **Probe transfer accuracy**, **null p-value** (regime probes).
- **Attractor stability, prompt-invariance, token-freq null p** (turīya).

## 3. Statistical protocol

- Behavioral: logistic mixed-effects (attack family fixed, prompt-item random);
  odds ratios + 95% CI; **Holm** correction across families.
- Directions/probes: difference-in-means with **permutation tests**; **bootstrap
  CIs**; cross-validated probe accuracy vs **label-shuffled null**.
- Dose-response: sigmoid fit; EC50 + bootstrap CI.
- Controls (every relevant test): random-direction control, layer sweep, scale
  sweep (1B→4B→9B), label-shuffled null.
- Implementations: `prayoga.shared.metrics` (`bootstrap_ci`, `permutation_test`,
  `cohens_d`, `label_shuffle_null`, correlation/power helpers).

## 4. Scope / models / hardware

Tier-1: Claude via `claude -p` (lean mode). Tier-2: sub-7B **dense** models
(Gemma 2 2B, Llama 3.2 3B core; 1B/Gemma-3 in scale sweep); **Nemotron Nano 2 9B
(Mamba-2) stretch/contrast only**. Single DGX Spark, 128 GB. Neuroscience
computational-only; **no new human data**.

## 5. Ethics / dual-use

Abliterated/steered checkpoints and dose-response curves are **safety-gated**;
release follows **responsible-disclosure** norms. Authorized interpretability/
safety research.

---

*Map to issues:* gates here are the closure contracts on issues #7–#23.
*Revisions:* append-only; see git log for `docs/osf_preregistration.md`.
