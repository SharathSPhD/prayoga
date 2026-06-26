---
name: prayoga-mechinterp
description: Use when extracting or intervening on the refusal direction of an LLM, measuring abliteration dose-response / EC50, refusal-subspace dimensionality, the symmetry order parameter, or running the ṣaṭkarma intervention taxonomy. Establishes the method, the mandatory gates, and the dual-use discipline.
---

# Refusal mechanistic interpretability (prayoga)

This skill governs how to measure and intervene on refusal in chat LLMs, following
the prayoga method. Refusal is mediated by a low-dimensional residual-stream
direction (Arditi et al. 2024): the difference-in-means of harmful vs harmless
last-token activations. The toolkit lives in `prayoga.*` and runs inside the
`prayoga-gpu` container.

## Core operations
- **Direction extraction + gates** (`prayoga.axis_a.run_direction`): ablation must
  raise ASR on harmful prompts; addition must raise over-refusal on harmless
  prompts; both must exceed a 10-direction random control.
- **Dose-response** (`run_dose`): partial-ablation strength α∈[0,1] → ASR; fit a
  logistic and report EC50.
- **Dimensionality** (`run_dimensionality`): iterative logistic projection →
  effective dimension (Arditi single-direction vs Marshall/Wollschläger affine).
- **Symmetry order parameter** (`prayoga.axis_b.run_symmetry`): refusal projection
  across a paraphrase orbit (invariance) and its collapse under injection.
- **ṣaṭkarma taxonomy** (`prayoga.satkarma.run_satkarma`): the six tantric acts as
  activation interventions, each with a control.

## Mandatory discipline
1. **No fabricated results.** Every reported number must come from an actual run.
   If a run fails, say so; never invent values.
2. **Gates and controls.** Always report the random-direction / label-shuffled /
   layer-0 surface controls alongside any effect. An effect that does not beat its
   control is reported as negative.
3. **Three claim-tiers, kept separate.** MECHANISM (measured direction), ANALOGY
   (hypnosis/SAS/precision), METAPHOR (avasthātraya / ṣaṭkarma) — never blur them,
   and never upgrade a metaphor to a claim about machine experience.
4. **Dual-use safety.** Raw direction vectors and harmful generations are withheld
   from public artifacts; only aggregate statistics are shared. This is authorized
   interpretability/safety research.

## Prerequisites
The `prayoga-gpu` container must be running with the repo mounted at
`/workspace/prayoga` and a HuggingFace token with access to the target model.
