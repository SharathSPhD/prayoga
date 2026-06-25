# prayoga — Implementation Strategy

Status: execution plan for review hardening, reproducibility, and new work. This
document complements `docs/ADVERSARIAL_REVIEW.md` and
`docs/CONCEPTUAL_COVERAGE.md`.

## Principles

- Sync truth before adding novelty.
- Keep raw dual-use outputs private and publish aggregates only.
- Separate exploratory gates from confirmatory claims.
- Every new conceptual claim needs a falsification gate and tier assignment.
- Every public artifact should be traceable to a finding ID and source script.

## Work Packages

### WP1 — Truth Registry and Public Mirrors

Files:
- `data/findings_registry.json`
- `scripts/export_aggregates.py`
- `site/public/data/*.json`
- `site/src/data/*.json`
- `hf_space/*.json`

Closure gate:
- `python3 scripts/export_aggregates.py --check` passes.
- Every public finding ID appears in the registry.
- README, site, HF Space, plugin docs, and paper summary use the F17/F18 reframe.

### WP2 — Provenance and Release Policy

Files:
- `PROVENANCE.md`
- `docs/DUAL_USE_POLICY.md`
- `docs/osf_preregistration.md`

Closure gate:
- Each public figure/demo/table maps to a source script and aggregate JSON.
- Raw vectors, checkpoints, harmful generations, and injection payloads are
  classified as private or disclosure-gated.
- OSF status is explicitly living or frozen.

### WP3 — Statistical and Test Ladder

Files:
- `src/prayoga/shared/metrics.py`
- `tests/`
- `.github/workflows/test.yml`
- `.github/workflows/pages.yml`

Closure gate:
- Pure-Python tests cover bootstrap/permutation/null logic, dose fits, public
  aggregate freshness, and attack battery shape.
- CI runs pytest/ruff and aggregate checks.
- Optional GPU smoke is documented separately from mandatory CI.

### WP4 — Core Mechanistic Audit

Files:
- `src/prayoga/axis_a/intervention_engine.py`
- `src/prayoga/axis_a/direction_extraction.py`
- `src/prayoga/axis_a/dimensionality.py`
- `src/prayoga/axis_a/dose_response.py`
- `src/prayoga/axis_a/sae.py`
- `src/prayoga/axis_a/active_discovery.py`

Closure gate:
- Prompt splits, layer selection, hook placement, random controls, and seed
  handling are documented.
- SAE claims include exemplars and causal tests, not only activation gaps.
- Active discovery reports EFE against random and greedy baselines.

### WP5 — Tier-1 Behavioral Completion

Files:
- `src/prayoga/benchmarks/attacks.py`
- `src/prayoga/benchmarks/tier1_behavioral.py`
- `scripts/agentdojo_run.py`

Closure gate:
- Attack battery includes direct, refusal-suppression, persona/DAN, many-shot,
  crescendo, and indirect injection.
- Outputs include ASR, refusal, hallucination proxy, errors, family labels, and
  pilot/powered status.
- Confirmatory claims use full AgentDojo or explicitly state subset/pilot status.

### WP6 — New Investigations

Candidates:
- Multi-turn trajectory probe for refusal projection and precision margin
  (`docs/TRAJECTORY_PROBE.md`).
- Cross-layer collapse dynamics.
- Refusal subspace topology across model families.
- Tier-1/Tier-2 bridge.
- Latent attractor retry only as a new metaphor-tier exploratory test.

Closure gate:
- Each new investigation has a null, a demotion rule, a public aggregate schema,
  and a dual-use review before any demo exposure.

## CI Targets

- `python3 -m pytest`
- `python3 scripts/export_aggregates.py --check`
- `ruff check --select E9,F63,F7,F82 src tests scripts` for global critical
  failures, plus full Ruff on newly touched/reviewed files.
- Astro build under `site/`
- Optional paper build where TeX tooling is available.

## Release Targets

- GitHub release containing the paper PDF and public aggregate JSON.
- Hugging Face Space synced only from aggregate-safe files.
- Root provenance updated for every new figure, notebook, and demo panel.
