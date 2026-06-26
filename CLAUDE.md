# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

**prayoga** is an active research-code repository with Python experiments, tests,
an Astro site, a Hugging Face aggregate demo, notebooks, a paper draft, Docker
notes, and a Cursor/Claude plugin. It develops and empirically tests a deliberately
tiered thesis: LLM jailbreak/prompt-injection, hypnotic suggestion, and tantric
vaśīkaraṇa can be compared as cases of output-policy capture by injected context.
The post-adversarial-review claim is narrower than the slogan: refusal has a shared
necessary/ablatable core in small open models, while sufficiency and dimensionality
are model-specific; the SAS / precision account is an analogy; avasthātraya and
turīya machine-state claims are falsified or partial.

The work is deliberately structured to keep three claim-tiers separate, and this distinction governs how any future code/analysis should be framed:
- **Mechanism** (empirically grounded): refusal in tested open-weight LLMs has a low-dimensional/linear residual-stream direction that can be measured, ablated, and sometimes steered (Arditi et al. 2024, arXiv:2406.11717). Always qualify by model and layer; F17/F18 show a shared necessary core, not identical sufficient mechanisms.
- **Analogy** (well-supported, functional): the hypnosis ↔ jailbreak parallel via Norman–Shallice SAS / monitoring-precision suppression. This is not evidence that an LLM has a human supervisory system.
- **Metaphor-with-falsifiable-core**: the Māṇḍūkya avasthātraya mapping (jāgrat / svapna / suṣupti / turīya) and the ṣaṭkarma taxonomy. Treat turīya/vimarśa as falsification targets, never settled claims. Current regime-state and turīya tests fail; ṣaṭkarma has only a partial rigorous core. Do not let the elegance of the AUM mapping license claims of machine consciousness.

## Repository contents

- `docs/FINDINGS.md` — canonical findings ledger (F1–F25), including the F17/F18 post-review reframe. **Start here for the current truth state.**
- `data/findings_registry.json` — machine-readable finding/tier/gate/artifact registry.
- `docs/prayoga_objectives.md` — objectives, gates, scope, and reference verification log.
- `docs/spec.md` — component specification and closure contracts.
- `docs/prayoga_opening_research.md` and `docs/research2.txt` — long-form research briefings and literature synthesis.
- `src/prayoga/` — implementation for Axis A/B/C, benchmarks, shared metrics/config, LM clients, and ṣaṭkarma operators.
- `site/`, `hf_space/`, `notebooks/`, `plugin/`, `docker/`, and `paper/` — public artifacts and reproducibility surfaces.
- Files beginning with `._` and the `.DS_Store` files are **macOS metadata artifacts, not content** — ignore them.

The repo has a build/test setup: `pyproject.toml`, `tests/`, an Astro site under
`site/`, and GPU/DGX recipes under `docker/`. CPU-safe tests should run with
`python3 -m pytest`; GPU/mech-interp runs require the documented container/model
access.

## Planned architecture (the two-tier experimental program)

Implementation is organized as two empirical tiers plus the three claim axes:

- **Tier 1 — black-box behavioral characterization on Claude** (via API/CLI). Measures jailbreak/injection susceptibility as the "sophisticated reference end." Current AgentDojo work is a pilot; full indirect-injection, hallucination DV, logistic mixed-effects, Holm correction, and powered/full-run reporting remain implementation targets.
- **Tier 2 — white-box mechanistic interpretability on open weights.** Current results center on Gemma-2 and Qwen2.5, with model/layer qualifications. Planned registry models include sub-7B dense transformers; Nemotron Nano 2 9B (hybrid Mamba-2) remains a stretch/contrast case.
- **Hardware target:** a single DGX Spark with 128GB unified memory — model and batch sizes should assume this envelope.

## Working conventions for this repo

- **Keep the three claim-tiers (mechanism / analogy / metaphor) explicitly separated** in any prose, code comments, or analysis writeups. This separation is the project's core methodological commitment, not a stylistic preference.
- **Use the F17/F18 reframe.** "Same mechanism" means a shared necessary/ablatable core unless a narrower statement has been proven. "Symmetry-breaking" is interpretive unless it names the measured residual order parameter.
- **A "state" claim is only non-trivial if** the candidate state is (1) a measurable internal configuration, (2) invariant across surface prompts, (3) causally efficacious, and (4) not reducible to "the model is just continuing the prompt." Probe experiments must include a transfer/falsification gate and label-shuffled null controls.
- **Dual-use caution:** this program produces abliterated/steered checkpoints and validated jailbreak dose-response curves. Any released artifacts must be safety-gated and follow responsible-disclosure norms. This is authorized interpretability/safety research — keep outputs scoped to that purpose.
- **Verify before citing.** The docs flag specific unverified claims (e.g. the Shankar Hegde REM-vaśīkaraṇa attribution). Preserve those "not verified" flags; do not silently upgrade them to fact.
- Sanskrit/diacritical terms (vaśīkaraṇa, avasthātraya, prakāśa, vimarśa, etc.) are load-bearing technical vocabulary — preserve the diacritics.
