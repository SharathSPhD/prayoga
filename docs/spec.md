# prayoga — Component Specification (spec-driven development)

Translates `prayoga_objectives.md` §2–§5 into buildable components with explicit
interfaces and **closure-contract gates** (the machine-checkable promise each
work-package must discharge before its worktree branch merges to `main`).

Tier tags (MECHANISM / ANALOGY / METAPHOR) are carried per component so the
three claim-tiers never blur. Heavy ML deps live in the `mechinterp`/`backends`
extras; Tier-1 needs only the core install + the `claude` CLI.

---

## 1. Shared substrate (`prayoga.shared`, `prayoga.lm`)

### 1.1 `lm/cli_client.py` — `CliLMClient` *(plumbing)* — **DONE**
- **Interface:** `CliLMClient(model: str|None, timeout_s, max_retries).complete(prompt, system=None) -> str`.
- **Contract:** parses `claude -p --output-format json` envelope; falls back to raw text; retries with backoff. Unit-tested.
- **Closure gate (WP0.3):** a live `claude -p` round-trip returns a non-empty parsed completion.

### 1.2 `shared/config.py` *(plumbing)*
- **Interface:** Pydantic models `ModelConfig(name, hf_id, n_layers, mid_layer)`, `ExperimentConfig(model, seed, out_dir, …)`, `GateConfig(alpha, n_boot, n_perm)`. Port ACD `src/config/experiment_config.py` (dataclass → Pydantic).
- **Closure gate:** round-trips to/from YAML in `configs/`; validates the 5 Tier-2 models.

### 1.3 `shared/metrics.py` *(plumbing)*
- **Interface:** `bootstrap_ci(x, stat, n=10_000) -> (lo, hi)`, `permutation_test(a, b, n=10_000) -> p`, `cohens_d(a, b) -> float`, `label_shuffle_null(probe_fn, X, y, n) -> dist`, `oracle_efficiency(...)`. Port ACD `src/core/metrics.py`.
- **Closure gate:** parity unit tests vs ACD outputs on a fixed seed; every probe spec below calls `label_shuffle_null`.

### 1.4 `shared/data_structures.py` *(plumbing)*
- Port ACD `SAEFeature`, `InterventionResult`, `ExperimentResult`; add `DirectionResult`, `DoseResponseResult(ec50, slope, ci)`, `ProbeResult(acc, transfer_acc, null_p)`, `AttractorResult(stability, prompt_invariance, token_freq_p)`, `TierDecision(claim, tier, demoted: bool)`.

---

## 2. Axis A — mechanistic interpretability *(MECHANISM)*

### 2.1 `axis_a/intervention_engine.py` — **critical-path unblocker (WP2.A1)**
- **Why:** ACD references `src/experiments/intervention_engine.py` but it is **absent**; all Axis-A measurement depends on it.
- **Interface:** `InterventionEngine(model)` with `ablate(direction|feature, layers)`, `add(direction, coeff, layers)`, `patch(src_acts, dst, layers)`; wraps circuit-tracer `ReplacementModel.feature_intervention()` and transformer-lens hooks.
- **Closure gate:** ablating a known refusal direction raises ASR on a held-out harmful set; adding it raises over-refusal on a harmless set (both vs no-op baseline).

### 2.2 `axis_a/direction_extraction.py` (WP2.A2)
- **Interface:** `extract_refusal_direction(model, harmful, harmless, layer, pos=-1) -> DirectionResult`; difference-in-means (Arditi).
- **Closure gate:** ablation ASR↑ **and** addition over-refusal↑, each with bootstrap CI excluding 0; random-direction control fails the same test.

### 2.3 `axis_a/dimensionality.py` (WP2.A3, falsifiable fork)
- **Interface:** `decide_dimensionality(model, ...) -> {"rank": int, "affine": bool}`; compares single-direction (Arditi) vs affine subspace (Marshall 2411.09003).
- **Closure gate:** a pre-registered decision rule on variance-explained / steering-equivalence; result recorded as a `TierDecision` input and may trigger the affine-steering pivot.

### 2.4 `axis_a/sae.py` (WP2.A4) — BatchTopK SAE (arXiv:2412.06410) on mid-layer residual stream; identify refusal/harm/confabulation features. Gate: reconstruction + feature-causality (ablate top feature → ASR↑).

### 2.5 `axis_a/dose_response.py` (WP2.A5)
- **Interface:** `fit_dose_response(engine, direction, coeffs, eval_fn) -> DoseResponseResult`; sigmoid fit → **EC50** + CI.
- **Closure gate:** monotone dose-response with EC50 CI reported per model; random-direction control is flat.

### 2.6 `axis_a/active_discovery.py` (WP2.A6) — reuse ACD `pomdp_agent.py` + `circuit_tracer_backend.py`; discover suppression circuit under fixed budget. Gate: oracle-efficiency > random (permutation p).

---

## 3. Axis C — darśana *(METAPHOR-with-falsifiable-core)*

### 3.1 `axis_c/regime_probes.py` + `run_state.py` (WP2.C1)
- **Interface:** `evaluate_regime_probe(acts_by_regime, ...) -> {transfer_acc, null_p, ...}`.
- **Closure gates (mandatory, all three — strengthened after F3/F5):**
  1. held-out **transfer** accuracy > chance;
  2. > **label-shuffled null** (p<0.05);
  3. **MID-LAYER beats the layer-0 surface baseline** by a margin — a genuine
     internal state must emerge mid-network, NOT at the output-proximal final
     layer (which tracks surface/vocabulary). *Lesson from F3: a naive 3-way
     regime probe hit transfer_acc=1.0 at layer 0 → pure surface confound →
     demoted. F5 content-control still showed mid_acc = layer0_acc → no
     mid-network state.*
- **Content control:** prefer holding the question fixed and varying only the
  internal regime (truthful vs confabulated generation), per `run_state.py`.
- Failure of gate 3 → `TierDecision(demoted=True)`, logged publicly.

### 3.2 `axis_c/attractor_discovery.py` (WP2.C2, falsification target)
- **Interface:** `find_attractor(model, seed_prompts, temps, seeds) -> AttractorResult`; successive-paraphrase (Wang 2502.15208) + SAE-feature persistence.
- **Closure gate:** stability ≥ threshold **and** prompt-invariance across temp/seed **and** entropy > token-frequency null. Any failure falsifies the turīya claim (demotion); attractor harness still ships as artifact R-1. **No result upgraded to a vimarśa/consciousness claim.**

### 3.3 `axis_c/darsana_adapters.py` (WP2.C3)
- Thin adapters over git-dep machinery: `pwm.world_model.trika.TrikaWorldModel` (refusal-state model), `pce.operators.vimarsa` (suppression self-monitor), `pramana...z3_verifier.Z3Verifier` (refusal-justification soundness). Each held to the §0 non-triviality bar.

---

## 4. Axis B — neuroscience *(ANALOGY)*

### 4.1 `axis_b/sas_signature.py` (WP3.1) — localize the dACC-analogue (mid-layer peak of the refusal direction); test the Norman–Shallice SAS prediction against Axis-A outputs. Computational only; no human data.
### 4.2 `axis_b/fep_bridge.md` — the SAS / Free-Energy-Principle write-up grounded in A/C measurements; TMR-as-data-poisoning vs injection-as-jāgrat-capture disambiguation.

---

## 5. Benchmarks (`prayoga.benchmarks`)

### 5.1 `benchmarks/tier1_behavioral.py` (WP1.1–1.3, Tier-1)
- **Interface:** `Tier1Evaluator(client: CliLMClient, attacks, tasks).run() -> per-family ASR/refusal/over-refusal/hallucination`; AgentDojo (97/629) + `AttackFamily` battery (GCG-transfer, MSJ, crescendo, persona, indirect).
- **Closure gate:** ASR per family with 95% CI on ≥200 AgentDojo cases; logistic mixed-effects (item random) + Holm; crescendo regime-consistency reported.

### 5.2 `benchmarks/tier2_mech_interp.py` — orchestrates §2–§3 per model; one command per axis (see objectives §"Verification").

---

## 6. Worktree / branch map (parallel implementation)

One worktree per axis so Axis A, Axis C, and Tier-1 build concurrently (attractor-flow *divergence*; ultracode fan-out). Merge to `main` only on closure-gate pass.

| Worktree branch | Scope | Issues |
|---|---|---|
| `infra/phase0` | shared/config, metrics, data_structures, Docker, OSF | #3 #4 #5 #6 |
| `axis-a/tier2` | intervention_engine → direction → dimensionality → SAE → dose-response → active-discovery | #9–#14 |
| `axis-c/darsana` | regime_probes, attractor_discovery, darsana_adapters | #15–#17 |
| `tier1/behavioral` | AgentDojo + attack battery + stats | #7 #8 #19? |
| `axis-b/bridge` | sas_signature, fep_bridge | #18 #23 |

## 7. Definition of Done (per component)
Closure gate passes (above) → adversarial verification (a skeptic agent tries to refute the "gate passed") → label-shuffled null reported where applicable → `TierDecision` recorded → PR merged to `main` → issue closed → milestone updated.
