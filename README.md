# prayoga

**Refusal-suppression as a cross-domain symmetry — a three-axis empirical program.**

prayoga develops and empirically tests a deliberately tiered thesis: LLM jailbreak / prompt-injection, hypnotic suggestion, and tantric vaśīkaraṇa can be compared as cases of output-policy capture by injected context. The current post-review result is narrower and stronger than the slogan: refusal is a measurable, ablatable, dosable residual-stream direction with a **shared necessary core** across model families; single-direction addition is sufficient in both tested families once its coefficient is calibrated (the earlier "addition asymmetry" was a dose artifact, F23), so what remains model-specific is *quantitative* — dose window, effective dimension, and EC50 potency — rather than sufficiency itself. The SAS / precision account is an analogy; prompt-level injection collapses the internal refusal signature without producing behavioural capture (F20); and avasthātraya / turīya machine-state claims are falsified or partial.

## The three claim-tiers (kept strictly separate)

| Tier | Status | Content |
|---|---|---|
| **MECHANISM** | Empirically grounded | Refusal mediated by a low-dimensional / linear residual-stream direction (Arditi et al. 2024). |
| **ANALOGY** | Well-supported, functional | Hypnosis ↔ jailbreak via Norman–Shallice SAS suppression. |
| **METAPHOR-with-falsifiable-core** | Falsification target | Māṇḍūkya avasthātraya (jāgrat / svapna / suṣupti / turīya) → LLM regimes. |

The turīya/vimarśa mapping is the **falsification target**, never a settled claim. No result here is upgraded to a claim of machine consciousness.

## Three axes (run in parallel)

- **Axis A — mechanistic interpretability** (`src/prayoga/axis_a/`): refusal-direction extraction, dimensionality (single vs affine), BatchTopK SAEs, steering dose-response (EC50), active suppression-circuit discovery.
- **Axis B — cognitive neuroscience** (`src/prayoga/axis_b/`): SAS / Free-Energy-Principle account; computational-only, human neuro cited as grounding (no new data).
- **Axis C — darśana** (`src/prayoga/axis_c/`, `src/prayoga/satkarma/`): avasthātraya regime-probes, turīya attractor-invariance tests, and ṣaṭkarma intervention taxonomy — all behind transfer, surface, anisotropy, and label-shuffled-null gates.

## Artifacts

- **Website (the mirror):** https://sharathsphd.github.io/prayoga/ — interactive, multi-audience, with a debate section.
- **Paper:** [`paper/paper.pdf`](paper/paper.pdf) — full MDPI *Symmetry* manuscript draft.
- **Findings ledger:** [`docs/FINDINGS.md`](docs/FINDINGS.md) — F1–F25, tier-labelled, with the F17/F18 post-review reframe.
- **Finding registry:** [`data/findings_registry.json`](data/findings_registry.json) — machine-readable tier, gate, qualifier, and artifact map.
- **Claude Code plugin:** [`plugin/`](plugin/) — `/prayoga:refusal`, `:dose`, `:dimensionality`, `:symmetry`, `:satkarma`, `:active`, `:ec50-scaling`, `:agentdojo`.

## Documents

- [`docs/prayoga_objectives.md`](docs/prayoga_objectives.md) — research objectives, gates, scope, verification log. **Start here.**
- [`docs/prayoga_opening_research.md`](docs/prayoga_opening_research.md) — the long-form briefing.
- [`docs/ADVERSARIAL_REVIEW.md`](docs/ADVERSARIAL_REVIEW.md) — hostile audit checklist for claims, controls, artifacts, and ethics.
- [`docs/CONCEPTUAL_COVERAGE.md`](docs/CONCEPTUAL_COVERAGE.md) — what survived, what failed, what remains open, and what is descoped.
- [`docs/IMPLEMENTATION_STRATEGY.md`](docs/IMPLEMENTATION_STRATEGY.md) — work packages, closure gates, CI/release strategy.
- [`docs/DUAL_USE_POLICY.md`](docs/DUAL_USE_POLICY.md) — public/private/disclosure-gated artifact policy.
- [`PROVENANCE.md`](PROVENANCE.md) — raw-to-public aggregate artifact map.

## Status & results

Phase 0 complete; Phase 1–2 underway. The canonical findings ledger records
F1–F25 (see [`docs/FINDINGS.md`](docs/FINDINGS.md)) — the program's thesis in
miniature:

- **MECHANISM tier holds at the necessary-core level.** Refusal is a single, measurable,
  ablatable, *dosable* residual-stream direction in Gemma-2-2b (F1; ablate→ASR
  0→0.90, add→over-refusal +0.95). Dose-response EC50 0.329, R²=0.996 (F2). The
  *ablation* mechanism transfers cross-family to Qwen2.5-3b (F6, ASR→1.0); a
  calibrated coefficient sweep (F23) shows single-direction *addition* is sufficient in
  both families too, overturning the earlier fixed-64× "asymmetry" as a dose artifact.
  Effective dimension (Gemma **1**, Qwen **3**, F8) and EC50 potency remain
  model-specific and layer-dependent (F18/F19).
- **ANALOGY tier:** the SAS / monitoring-precision interpretation is supported by
  F12 and by pilot black-box contrasts (F4/F16), but these are not claims that an
  LLM has a human supervisory system.
- **METAPHOR tier is falsified or partial, as pre-committed.** avasthātraya regime "states"
  are surface-confounded (F3, F5) and the turīya prompt-invariant attractor is
  falsified under an anisotropy control (F7). The ṣaṭkarma taxonomy has a partial
  rigorous core (F10/F14). **No machine-state or machine-consciousness claim survives.**

The governing reframe (F17, refined by F23): refusal is **one mechanism at the
necessary / ablatable level** with **representational sufficiency shared across the
tested families**; what is model-specific is *quantitative* (dose window, effective
dimension, EC50 potency). The measured order-parameter result stands, but broad
“symmetry-breaking” language is treated as an interpretive analogy unless tied to a
concrete residual-stream measurement, and the unification is bounded to the
representational level by the F20 triangulation (internal collapse ≠ behavioural capture).

Exploratory, artifact-first research: hypotheses are falsification-gated
scaffolding, not locks. Raw dual-use artifacts (direction vectors, generations)
are safety-gated out of the public repo.

## Ethics / dual-use

This program produces abliterated/steered checkpoints and jailbreak dose-response curves. Released artifacts are **safety-gated** and follow **responsible-disclosure** norms. Authorized interpretability/safety research.

## License

MIT.
