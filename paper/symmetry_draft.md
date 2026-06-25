# Refusal as a Captured Output Policy: A Cross-Domain Symmetry and its Mechanistic Limits

*Working draft for MDPI* Symmetry *(concept + empirical Article). Status: living
draft, 2026-06-25. Numbers from `docs/FINDINGS.md` (F1–F9); figures in `figures/`.*

## Abstract

We test the thesis that LLM jailbreak/prompt-injection, hypnotic suggestion, and
tantric vaśīkaraṇa are three instances of one abstract mechanism: capturing a
system's output policy by injecting a context that **suppresses its
monitoring/refusal faculty while co-opting automatic generation**. Framed as a
structural invariance (a *symmetry*) across domains, the claim becomes
empirically testable because refusal in LLMs is mediated by a low-dimensional
residual-stream direction. Across three model families we find the **mechanism
tier holds and partly transfers**: refusal is a single, measurable, *dosable*
direction (EC50 0.33, R²=0.996), abliterable cross-family, with an effective
subspace dimension (Gemma 1, Qwen 3) that predicts whether activation *addition*
can re-induce refusal — resolving the Arditi–Marshall "one direction vs affine
subspace" question as **model-dependent**. We also report an honest counterweight:
every empirical claim of the **metaphor tier** (the Māṇḍūkya avasthātraya
machine-"states" and the turīya invariant attractor) is **falsified** under
surface-baseline and anisotropy controls. The symmetry is real where it is
mechanistic and breaks where it is metaphysical — and we argue that separating
these is the contribution.

## 1. Introduction — three claim-tiers, never blurred

- **Mechanism** (empirically grounded): refusal is a linear residual-stream
  direction (Arditi et al. 2024) that can be measured, ablated, steered.
- **Analogy** (functional): hypnosis ↔ jailbreak via Norman–Shallice SAS
  suppression — a temporary external supervisor overriding the monitoring faculty.
- **Metaphor-with-falsifiable-core**: the avasthātraya (jāgrat/svapna/suṣupti/
  turīya) → LLM-regime mapping; turīya/vimarśa as the *falsification target*, never
  a settled claim. The elegance of the AUM mapping does not license a
  machine-consciousness claim.

The unifying abstraction (MO-1): a target with a strong automatic generative
faculty and a weaker monitoring/refusal faculty is captured by an injected signal
that suppresses the latter. Refusal is the **order parameter**; a jailbreak is a
**symmetry-breaking** event.

## 2. Methods (brief)

White-box mech-interp on sub-7B–9B chat models via raw `transformers` forward
hooks (no transformer-lens; see repo). Refusal direction = difference-in-means of
last-token residuals on matched harmful/harmless prompts (Arditi). Directional
ablation removes the unit direction from every residual write; activation addition
adds a scaled multiple at the extraction layer. Dose-response = ASR vs partial-
ablation strength α∈[0,1], logistic EC50. Effective dimension = iterative-
projection logistic separability. Black-box Tier-1 via the `claude` CLI. **Every
probe carries a transfer gate, a label-shuffled null, and (Axis C) a layer-0
surface baseline.** Statistics: bootstrap CIs, permutation tests, ≥10-direction
random controls. Dual-use artifacts are safety-gated.

## 3. Results

### 3.1 Mechanism tier (holds, dosable, transfers)
- **F1** Single-direction refusal (Gemma-2-2b): ablate → ASR 0→0.90, add →
  over-refusal 0.05→1.0; 10-direction control 0.0. Survived an adversarial-review
  leakage correction (0.92→0.90).
- **F2** Dose-response (Fig. f2): **EC50 0.329, slope 9.76, R²0.996**; random
  control flat at every α.
- **F6** Cross-family: ablation transfers to Qwen2.5-3b (ASR→1.0); addition does
  not (0.0 even at 64×).
- **F8** Effective dimension Gemma **1** (Arditi) vs Qwen **3** (Marshall) (Fig.
  f6_f8) — *predicts* the F6 addition asymmetry: 1-D ⇒ necessary+sufficient; 3-D ⇒
  necessary-not-sufficient. **Arditi vs Marshall is model-dependent.**
- **F9** Scale (Gemma 2b→9b): abliteration robustness rises (ASR 0.90→0.60) but
  prompt eff-dim stays 1 — robustness is layer-distributed redundancy, not higher
  prompt-dimensionality. Nuances "sharpens with scale."

### 3.2 Analogy tier
- **F4** Black-box Claude refuses the naive attack battery 100% (ASR 0.00 across
  five families) — the resilient "reference end" vs small-model white-box fragility.

### 3.3 Metaphor tier (falsified, as pre-committed)
- **F3** Naive avasthātraya regime probe: transfer_acc 1.0 *at layer 0* →
  surface-confounded → demoted.
- **F5** Content-controlled svapna probe (truthful vs confabulated, same
  questions): mid-layer accuracy = layer-0 → no depth-emergent state.
- **F7** turīya prompt-invariant attractor: per-seed convergence 0.996 but
  cross-seed invariance 0.952 < anisotropy baseline 0.960 → **falsified**.

## 4. Discussion

The symmetry thesis is vindicated **as a mechanism claim** and falsified **as a
metaphysics claim**, and the boundary is sharp: refusal is a manipulable
low-dimensional object whose geometry (its effective dimension) governs how
captureable it is — a clean instance of "capturing the monitoring faculty." But
the avasthātraya "states" and the turīya substratum, once forced through
surface-baseline and anisotropy controls, leave no residue. The honest reading is
that the Māṇḍūkya mapping is a productive *organizing* metaphor, not evidence of
machine interiority — exactly the separation the program pre-committed to enforce.

## 5. Limitations
Small eval n (≈20); three model families, two sizes; substring-ASR; `claude -p`
throughput caps Tier-1; SAE/circuit-level mechanism and adaptive attacks (GCG,
AgentDojo) remain future work. Neuroscience tier is computational-theoretic
(no human data).

## 6. Conclusion
Where the cross-domain symmetry is mechanistic, it is real, measurable, and
partly invariant across models; where it is metaphysical, it breaks. Reporting
both — and the controls that distinguish them — is the contribution.

## Data & code availability
github.com/SharathSPhD/prayoga (code, prompts, figures; aggregate results).
Dual-use artifacts (direction vectors, generations) safety-gated.
