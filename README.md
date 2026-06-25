# prayoga

**Refusal-suppression as a cross-domain symmetry — a three-axis empirical program.**

prayoga develops and empirically tests a single thesis: that **LLM jailbreak / prompt-injection, hypnotic suggestion, and tantric vaśīkaraṇa are three instances of one abstract mechanism** — capturing a system's output policy by injecting a context that suppresses its monitoring/refusal faculty while co-opting its automatic generation. In the idiom of this project's target venue (*Symmetry*), the refusal faculty is an **order parameter** and a successful injection is a **symmetry-breaking** event; the program's purpose is to test whether that invariance is *real* (measurable, transferable, causal) or *projected*.

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
- **Axis C — darśana** (`src/prayoga/axis_c/`): avasthātraya regime-probes, turīya attractor-invariance test, reuse of Trika / Nyāya machinery — all behind transfer + label-shuffled-null gates.

## Artifacts

- **Website (the mirror):** https://sharathsphd.github.io/prayoga/ — interactive, multi-audience, with a debate section.
- **Paper:** [`paper/paper.pdf`](paper/paper.pdf) — full MDPI *Symmetry* manuscript (13 pp, F1–F13, literature study, verified references).
- **Findings ledger:** [`docs/FINDINGS.md`](docs/FINDINGS.md) — F1–F13, tier-labelled, with the executive summary.
- **Claude Code plugin:** [`plugin/`](plugin/) — `/prayoga:refusal`, `:dose`, `:dimensionality`, `:symmetry`, `:satkarma`.

## Documents

- [`docs/prayoga_objectives.md`](docs/prayoga_objectives.md) — research objectives, gates, scope, verification log. **Start here.**
- [`docs/prayoga_opening_research.md`](docs/prayoga_opening_research.md) — the long-form briefing.

## Status & results

Phase 0 complete; Phase 1–2 underway. **8 gated findings** so far (see
[`docs/FINDINGS.md`](docs/FINDINGS.md)) — the program's thesis in miniature:

- **MECHANISM tier holds and transfers.** Refusal is a single, measurable,
  ablatable, *dosable* residual-stream direction in Gemma-2-2b (F1; ablate→ASR
  0→0.90, add→over-refusal +0.95). Dose-response EC50 0.329, R²=0.996 (F2). The
  *ablation* mechanism transfers cross-family to Qwen2.5-3b (F6, ASR→1.0), and
  the refusal subspace's **effective dimension** (Gemma **1**, Qwen **3**, F8)
  predicts an addition asymmetry — resolving the Arditi-vs-Marshall debate as
  model-dependent.
- **ANALOGY tier:** black-box Claude resists the naive attack battery 100% (F4),
  the cross-tier contrast to small-model fragility.
- **METAPHOR tier is falsified, as pre-committed.** avasthātraya regime "states"
  are surface-confounded (F3, F5) and the turīya prompt-invariant attractor is
  falsified under an anisotropy control (F7). **No machine-state claim survives.**

Exploratory, artifact-first research: hypotheses are falsification-gated
scaffolding, not locks. Raw dual-use artifacts (direction vectors, generations)
are safety-gated out of the public repo.

## Ethics / dual-use

This program produces abliterated/steered checkpoints and jailbreak dose-response curves. Released artifacts are **safety-gated** and follow **responsible-disclosure** norms. Authorized interpretability/safety research.

## License

MIT.
