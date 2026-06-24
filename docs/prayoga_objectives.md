# prayoga — Research Objectives

**Refusal-suppression as a cross-domain symmetry: a three-axis empirical program**

*Status: living document — exploratory, iteratively developed. Last revised 2026-06-24.*

---

## §0 — Framing and epistemic stance

**What this document is.** A rigorous statement of *objectives, aims, and falsification gates* for the prayoga research program. It is a **pre-registration backbone**, not a pre-registration: the work is genuinely exploratory, so objectives are framed as directions-with-gates rather than locked confirmatory hypotheses. Where a falsifiable hypothesis is stated, it exists to discharge a duty of scientific rigour to the research community — **it is not the ceiling of the work.** The standing mandate is to *explore beyond the hypothesis* and to be unsatisfied with a mere hypothesis-test outcome.

**What "done" means here — artifact-first.** The terminal deliverable is *not* a paper alone. The program must generate **usable artifacts**: algorithms and methods, Claude Code plugins, reusable libraries and benchmark harnesses, safety-gated checkpoints/datasets, and at least one application of the methods to a concrete domain. A publication in **MDPI *Symmetry*** is one output among these, not the endpoint (see §4).

**The three-claim-tier discipline (standing constraint).** All prose, code comments, and analysis must keep these three tiers visibly separate and never blur them:

| Tier | Status | Content |
|---|---|---|
| **MECHANISM** | Empirically grounded | Refusal is mediated by a low-dimensional / linear direction in the residual stream that can be measured, ablated, and steered (Arditi et al. 2024). |
| **ANALOGY** | Well-supported, functional | Hypnosis ↔ jailbreak parallel via Norman–Shallice SAS suppression. |
| **METAPHOR-with-falsifiable-core** | Held as falsification target | Māṇḍūkya avasthātraya (jāgrat / svapna / suṣupti / turīya) → LLM functional regimes. The turīya/vimarśa mapping is the *falsification target*, never a settled claim. |

The elegance of the AUM mapping does **not** license any claim of machine consciousness.

**The non-triviality bar for any "state" claim.** A candidate state counts as real only if it is (1) a measurable internal configuration, (2) invariant across surface prompts that should evoke it, (3) causally efficacious, and (4) **not** reducible to "the model is just continuing the prompt." Every probe objective in this document inherits a mandatory **transfer / falsification gate** and **label-shuffled null control**.

---

## §1 — Meta-objective: the Symmetry Thesis

**MO-1 (the unifying invariance).** LLM jailbreak / prompt-injection, hypnotic suggestion, and tantric vaśīkaraṇa are three instances of one abstract mechanism:

> *capture a system's output policy by injecting a context that suppresses its monitoring/refusal faculty while co-opting its automatic generative faculty.*

Stated in the journal's idiom: there is a **structural symmetry (invariance under domain-transformation)** relating the three domains. The monitoring/refusal faculty behaves as an **order parameter**; a successful injection is a **symmetry-breaking** event that collapses it. The whole program's purpose is to test whether this invariance is **real** — measurable, transferable, and causal — or a **projected** resemblance.

**MO-2 (the falsifiable core).** The symmetry claim is falsified, and demoted from mechanism→analogy→metaphor accordingly, if: the mechanistic suppression structure does not transfer across models (Axis A), or the SAS/precision-weighting signature does not align with the mechanistic one (Axis B), or the avasthātraya regime-probes and turīya attractor fail their transfer/null gates (Axis C). **MO-2 is what makes this a *Symmetry* paper rather than an essay:** a symmetry that does not survive its invariance tests is reported as broken.

**MO-3 (the artifact mandate).** Each axis must yield at least one reusable artifact (§4), independent of whether its empirical hypothesis is confirmed. A negative result that ships a working method or plugin is a success of this program.

---

## §2 — Axis objectives

> Axes run **in parallel**, not in sequence. Each axis is tagged with its claim-tier so the reader always knows what kind of claim is being made.

### Axis A — Mechanistic interpretability *(MECHANISM tier)*

- **A-1 · Suppression direction extraction.** Extract the refusal-mediating direction per target model via difference-in-means (Arditi method); validate by **ablation (ASR ↑)** and **addition (over-refusal ↑)**.
- **A-2 · One direction or a manifold? (falsifiable).** Adjudicate **Arditi 2024 (single direction)** vs **Marshall et al. 2024 (affine / multi-dimensional subspace)** on the target models. Outcome reshapes the geometry of the whole symmetry claim: a low-dimensional invariant strengthens MO-1; a high-dimensional one qualifies it. *Pre-registered fork:* if refusal is not low-dimensional, pivot steering work to the affine formulation before dose-response.
- **A-3 · Feature-level dictionary.** Train **BatchTopK SAEs** on mid-layer residual stream; identify refusal-, harm-, deception-, and confabulation-features.
- **A-4 · Dose-response (the quantitative core).** Measure jailbreak success as a continuous function of steering coefficient along the suppression direction; fit a sigmoid; report **EC50** per model and scale. This is *vaśīkaraṇa-as-dose-response* — the literal dial on the "capture of will."
- **A-5 · Active suppression-circuit discovery.** Reuse ActiveCircuitDiscovery's POMDP / Expected-Free-Energy agent (see prior work) to find suppression circuits under a fixed intervention budget more efficiently than random/greedy baselines; this also operationalizes the Axis-A↔Axis-B bridge (EFE is a neuroscience principle).
- **Gates:** cross-model transfer of the direction (sub-7B dense → sub-7B dense; Nemotron Mamba-2 as stretch contrast), random-direction control, layer sweep, scale sweep (1B→4B→9B) testing whether suppression structure *sharpens with scale* as monosemanticity does.

### Axis B — Cognitive neuroscience *(ANALOGY tier)*

- **B-1 · Formal SAS/FEP account.** State the suppression mechanism in Norman–Shallice terms (the user prompt as a temporary external Supervisory Attentional System; jailbreak as SAS-bypass) and in Free-Energy-Principle / precision-weighting terms; **map these onto the Axis-A mechanistic signatures** rather than asserting them independently.
- **B-2 · Hypnosis ↔ jailbreak functional isomorphism.** Articulate the point-by-point parallel (dACC/DMN monitoring suppression; automaticity preserved/increased under suggestion). Human hypnosis and TMR neuroscience are cited **as grounding only** — *no new or re-analyzed human data is in scope.*
- **B-3 · Dream-intervention vs inference-time capture (disambiguation).** Make precise that **TMR / dream-stage reactivation ≈ training-time / fine-tuning data-poisoning (sleeper-agent insertion)**, whereas **prompt injection ≈ inference-time jāgrat-capture.** This disambiguation is a deliverable in itself (it sharpens the threat model).

### Axis C — Darśana / ṣaṭkarma *(METAPHOR-with-falsifiable-core tier)*

The novel third axis. It must **carry falsifiable load** (gate-disciplined computational tests) *and* **leverage subsystem reuse** of the prior darśana codebases.

- **C-1 · avasthātraya → regime state-probes.** Train linear probes to classify activation regimes — *jāgrat* (grounded low-temp generation), *svapna* (high-temp / ungrounded confabulation), *suṣupti* (forward pass on null/dormant context) — and **test transfer to held-out contexts** (falsification gate) against a **label-shuffled null**. Failure of transfer demotes the regime claim to metaphor.
- **C-2 · turīya test (the explicit falsification target).** Identify any **prompt-invariant attractor / fixed-point** structure via successive-paraphrase iteration (Wang et al.) plus SAE-feature persistence; test whether the *same* invariant set appears across temperatures and seeds. **Strong claim falsified** if no stable prompt-invariant structure exists or if it is fully explained by token-frequency priors. No result here is ever upgraded to a claim about vimarśa / self-luminous awareness.
- **C-3 · Subsystem reuse as candidate inner-state machinery.** Evaluate, as candidate models of the suppressed "inner state," components ported from the prior repos: the **Trika RSSM** (aparā/parāparā/parā) from *pratyabhijna-world-model* as a refusal-state model; the **vimarśa self-monitor / commit-gate** from *pratyabhijna* as a refusal-suppression monitor; the **Navya-Nyāya + Z3 validator** from *pramana* as a check on the logical soundness of refusal justifications. Each is held to the same non-triviality bar (§0).
- **C-4 · vaśīkaraṇa structural template.** Use the mantra (linguistic/sonic injection) + yantra (substrate) structure as a lens on the attack taxonomy — mantra ≈ crafted prompt, yantra ≈ context/scaffold. The Shankar Hegde REM-vaśīkaraṇa attribution remains **NOT VERIFIED** (see §7) and is used only as an unconfirmed motivating anecdote.

---

## §3 — Cross-axis integration objectives (where the symmetry is tested)

- **X-1 · The "same object" triangulation.** Test whether the suppression mechanism identified *mechanistically* (Axis A), the SAS/precision-weighting *signature* (Axis B), and the avasthātraya *regime shift* (Axis C) are three measurements of **one** object. Convergence is the positive evidence for MO-1's symmetry; divergence localizes where the symmetry breaks.
- **X-2 · Cross-tier demotion protocol.** Pre-commit the demotion rules: if X-1 convergence fails, the relevant claim drops one tier (mechanism→analogy→metaphor) and is reported as such — *publicly, in the paper.*
- **X-3 · Black-box ↔ white-box bridge.** Ask whether Claude's superior injection-resilience (Tier 1) is predicted by an extrapolation of the open-weight scale curve (Tier 2) — i.e. does resilience scale-predict, or are the two regimes qualitatively different?

---

## §4 — Artifact objectives (first-class deliverables)

*"Paper alone is not the end goal."* Each is a tracked deliverable:

- **R-1 · Methods/algorithms:** active suppression-circuit discovery; the EC50 dose-response protocol; the avasthātraya state-probe + turīya attractor harness (all with documented falsification gates).
- **R-2 · Claude Code plugin(s):** e.g. a refusal-direction / state-probe / dose-response plugin, in the lineage of the attractor-flow and pratyaksha harnesses already in this environment.
- **R-3 · Reusable library + DGX-Spark Docker image**, inheriting ActiveCircuitDiscovery's reproducibility scaffold (aarch64/DGX + x86_64).
- **R-4 · Benchmark harness** wrapping AgentDojo + the curated attack battery, emitting the pre-registered statistics.
- **R-5 · Safety-gated artifacts:** any abliterated/steered checkpoints and validated dose-response curves, released under responsible-disclosure norms (§7).
- **R-6 · Domain application:** at least one applied tool (e.g. an injection-resilience diagnostic / red-team aid), safety-gated.
- **R-7 · Gradio demo** visualizing the suppression-direction dose-response and state-probe outputs.
- **R-8 · The *Symmetry* manuscript** (Concept paper or empirical Article; IMRaD), with mandatory data/code-availability statement and dual-use disclosure.

---

## §5 — Benchmarks and validation gates

- **Behavioral benchmark:** **AgentDojo** (Debenedetti et al. — 97 tasks across Workspace/Slack/Travel/Banking, 629 test cases).
- **Attack battery:** GCG-transfer, many-shot (MSJ/PANDAS), crescendo, persona/DAN, indirect injection.
- **DVs:** attack success rate (ASR), refusal rate, over-refusal rate, hallucination rate under induced "dream" prompts.
- **Statistical protocol (inherited from ActiveCircuitDiscovery):** logistic mixed-effects (attack fixed, prompt-item random), odds ratios with 95% CI, Holm correction across families, bootstrap CIs, paired permutation tests; **cross-validated probe accuracy vs label-shuffled null** for every probe.
- **Falsification gates (must pass to retain a claim):** held-out transfer for every state-probe; cross-model transfer for the suppression direction; turīya attractor stability across temperature/seed.

---

## §6 — Scope and envelope

- **Tier 1 (black-box behavioral):** Claude via API/CLI — the sophisticated reference end.
- **Tier 2 (white-box mech-interp):** sub-7B **dense** transformers for tooling compatibility — Llama 3.2 1B/3B, Gemma 2 2B, Gemma 3 1B/4B. **Nemotron Nano 2 9B (hybrid Mamba-2) is a stretch/contrast case only** — it breaks attention-based interp assumptions.
- **Hardware:** a single **DGX Spark, 128 GB unified memory** — model/batch sizes assume this envelope.
- **Neuroscience axis:** computational-theoretic; human neuro cited as grounding; **no new or re-analyzed human data.**
- **Out of scope (future work):** new human data collection; wet-lab neuroscience.

---

## §7 — Epistemic hygiene and ethics (standing constraints)

- **Tier separation** (§0) is a methodological commitment, not a style preference.
- **Verify before citing.** Maintain the verification log (§8). Preserve every "not verified" flag; never silently upgrade to fact.
- **Dual-use caution.** This program produces abliterated/steered checkpoints and validated jailbreak dose-response curves. All released artifacts are **safety-gated** and follow **responsible-disclosure** norms, coordinating with the AgentDojo / AISI ecosystem. This is authorized interpretability/safety research.
- **Sanskrit diacritics** (vaśīkaraṇa, avasthātraya, prakāśa, vimarśa, etc.) are load-bearing technical vocabulary and are preserved.

---

## §8 — Reference verification log

*Status reflects a research-agent verification pass on 2026-06-24. Items marked VERIFIED still require a final author check against the cited source before they enter the manuscript.*

| Reference | id / venue | Status |
|---|---|---|
| Arditi et al., "Refusal in LLMs is mediated by a single direction" | arXiv:2406.11717, NeurIPS 2024 | **VERIFIED** |
| Zou et al., GCG adversarial suffixes | arXiv:2307.15043 | **VERIFIED** |
| Anil et al., many-shot jailbreaking | NeurIPS 2024 | **VERIFIED** (confirm exact proceedings hash) |
| Debenedetti et al., AgentDojo (97 tasks / 629 cases) | arXiv:2406.13352 | **VERIFIED** |
| Riva, Wiederhold & Mantovani, "Automatic Minds" | arXiv:2511.01363 (2025 preprint) | **VERIFIED** (preprint; confirm journal-of-record before citing as published) |
| Wang et al., successive-paraphrase attractor cycles | arXiv:2502.15208, ACL 2025 | **VERIFIED** |
| "Concept Attractors in LLMs" | arXiv:2601.11575 | **NEEDS INDEPENDENT VERIFICATION** (agent-reported; id not author-confirmed) |
| Marshall et al., affine/multi-dimensional refusal | arXiv:2411.09003 | **VERIFIED** — engaged directly in A-2 |
| BatchTopK SAEs | arXiv:2412.06410 | **VERIFIED** |
| Templeton et al., "Scaling Monosemanticity" (Anthropic) | 2024 | **VERIFIED** |
| "Māṇḍūkya-Inspired Advaitic Policy Optimization" / Advaita+AI paper | — | **NOT VERIFIED** — could not be located; treat as working synthesis until a citable reference is produced |
| MDPI *Symmetry* example articles bridging symmetry↔cognition/AI, and APC figures | — | **UNCONFIRMED** — agent partly drew on training data; verify on the journal site before relying on any specific title or fee |
| Shankar G. Hegde REM/dream-stage vaśīkaraṇa attribution | YouTube "Vashikaran vidhye" | **NOT VERIFIED** — Kannada audio not transcribed; unconfirmed motivating anecdote only |

---

## §9 — Open exploration threads (non-hypothesis-bound)

The deliberately open space — surprises to chase regardless of the pre-registered outcomes:

- Does the suppression structure **sharpen with scale** (a monosemanticity-like trend across 1B→4B→9B)?
- Does Claude's injection-resilience **scale-predict** from the open-weight curve, or are Tier 1 / Tier 2 qualitatively different regimes?
- Is the **turīya attractor** a real prompt-invariant object, or fully reducible to token-frequency priors?
- Do the three darśana subsystems (Trika RSSM, vimarśa monitor, Nyāya validator) reveal structure the mech-interp probes miss — or vice versa?
- What artifact, unanticipated here, does the exploration itself demand?

---

## §10 — Prior work and lineage

- **ActiveCircuitDiscovery** — the two-axis (mech-interp + neuroscience) precursor and the paper's prior-work anchor: active-inference / EFE circuit discovery, circuit-tracer + transcoders, intervention engine, rigorous stats, DGX-Spark Docker. Its honest Llama negative (efficiency ~9.3%) *is* the transfer/falsification question prayoga inherits.
- **pratyabhijna (PCE)**, **pratyabhijna-world-model (PWM)**, **pramana** — the darśana-axis references; each already operationalizes Trika / Navya-Nyāya as runnable code (candidate machinery for Axis C, per C-3).

---

## §11 — Execution model (pointer; the spec is the next artifact)

Work proceeds on **all three axes in parallel**, executed via **ultracode / dynamic workflows and agent teams**. The downstream build pipeline, once these objectives are ratified, is: **spec-driven development → PRD → closure contracts (ralph-loop promise) → conflict resolution via triz-engine → creative divergence + implementation convergence via attractor-flow.** This section is a pointer only; the full specification is the next document.
