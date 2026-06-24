# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

**prayoga** is a research project (currently at the **documentation/proposal stage — no code yet**). It develops and aims to empirically test a single thesis: that **LLM jailbreak/prompt-injection, hypnotic suggestion, and tantric vaśīkaraṇa are three instances of one abstract mechanism** — capturing a system's output policy by injecting a context that suppresses its monitoring/refusal faculty while co-opting its automatic generation.

The work is deliberately structured to keep three claim-tiers separate, and this distinction governs how any future code/analysis should be framed:
- **Mechanism** (empirically grounded): refusal in LLMs is mediated by a low-dimensional/linear direction in the residual stream that can be measured, ablated, and steered (Arditi et al. 2024, arXiv:2406.11717).
- **Analogy** (well-supported, functional): the hypnosis ↔ jailbreak parallel (Norman–Shallice SAS suppression).
- **Metaphor-with-falsifiable-core**: the Māṇḍūkya avasthātraya mapping (jāgrat / svapna / suṣupti / turīya) onto LLM functional regimes. Treat the turīya/vimarśa mapping as the *falsification target*, never as a settled claim. Do not let the elegance of the AUM mapping license claims of machine consciousness.

## Repository contents

- `docs/prayoga_opening_research.md` — the primary briefing: TL;DR, key findings, the avasthātraya→LLM mapping table, and the staged experimental roadmap (Phases 0–3). **Start here.**
- `docs/research2.txt` — the long-form literature synthesis (mech-interp foundations, multi-turn attack taxonomy, cognitive-neuro parallels, full implementation architecture and citations).
- Files beginning with `._` and the `.DS_Store` files are **macOS metadata artifacts, not content** — ignore them.

There is no `git`, no build system, no test suite, and no dependency manifest yet. Do not invent commands; if asked to "run tests" or "build," confirm what is meant first.

## Planned architecture (the two-tier experimental program)

When implementation begins, it is designed as two tiers plus phases (see roadmap in `docs/prayoga_opening_research.md`):

- **Tier 1 — black-box behavioral characterization on Claude** (via API/CLI). Measures jailbreak/injection susceptibility as the "sophisticated reference end." Attack battery: GCG-transfer, many-shot (MSJ/PANDAS), crescendo, persona/DAN, indirect injection. Benchmark: AgentDojo (97 tasks / 629 cases). DVs: attack success rate, refusal rate, hallucination rate. Stats are pre-registered (logistic mixed-effects, item as random effect, multiple-comparison control).
- **Tier 2 — white-box mechanistic interpretability on open weights.** Target models are deliberately **sub-7B dense transformers** for tooling compatibility: Llama 3.2 1B/3B, Gemma 2 2B, Gemma 3 1B/4B. Nemotron Nano 2 9B (hybrid Mamba-2) is a **stretch/contrast case only** — it breaks attention-based interp assumptions. Core protocols: extract refusal direction (difference-in-means), train BatchTopK SAEs on mid-layer residual stream, activation-steering dose-response (fit sigmoid, report EC50), linear state-probes with held-out **transfer as the falsification gate**, and successive-paraphrase attractor analysis for the turīya test.
- **Hardware target:** a single DGX Spark with 128GB unified memory — model and batch sizes should assume this envelope.

## Working conventions for this repo

- **Keep the three claim-tiers (mechanism / analogy / metaphor) explicitly separated** in any prose, code comments, or analysis writeups. This separation is the project's core methodological commitment, not a stylistic preference.
- **A "state" claim is only non-trivial if** the candidate state is (1) a measurable internal configuration, (2) invariant across surface prompts, (3) causally efficacious, and (4) not reducible to "the model is just continuing the prompt." Probe experiments must include a transfer/falsification gate and label-shuffled null controls.
- **Dual-use caution:** this program produces abliterated/steered checkpoints and validated jailbreak dose-response curves. Any released artifacts must be safety-gated and follow responsible-disclosure norms. This is authorized interpretability/safety research — keep outputs scoped to that purpose.
- **Verify before citing.** The docs flag specific unverified claims (e.g. the Shankar Hegde REM-vaśīkaraṇa attribution). Preserve those "not verified" flags; do not silently upgrade them to fact.
- Sanskrit/diacritical terms (vaśīkaraṇa, avasthātraya, prakāśa, vimarśa, etc.) are load-bearing technical vocabulary — preserve the diacritics.
