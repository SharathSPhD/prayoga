# prayoga — Adversarial Review

Status: living hostile-review checklist. Canonical evidence lives in
`docs/FINDINGS.md`; machine-readable claim metadata lives in
`data/findings_registry.json`.

## Review Contract

The project should be reviewed as if the strongest-looking claim is wrong until
it survives controls. The reviewer should ask four questions of every artifact:

1. What claim-tier is being made?
2. What would falsify or demote it?
3. Is the public artifact using the same tier and qualifier as the ledger?
4. Could the artifact expose dual-use material or overstate a pilot result?

## Corrected Headline

The defensible post-review thesis is:

> Refusal has a shared necessary / ablatable core in tested open-weight models,
> while sufficiency and dimensionality are model- and layer-dependent. SAS /
> monitoring-precision is an analogy. avasthātraya and turīya machine-state
> claims are falsified; ṣaṭkarma survives only as a partial intervention taxonomy.

Broad "symmetry-breaking" language must be treated as interpretive unless it is
explicitly tied to the measured residual order parameter.

## Claim-Tier Audit

| Area | Hostile question | Required response |
|---|---|---|
| Mechanism | Does "single direction" name the model and layer? | Qualify by model/layer and cite F18 for depth dependence. |
| Mechanism | Does ablation transfer get misread as addition transfer? | Use "shared necessary core, model-specific sufficiency." |
| Analogy | Does SAS/precision language imply human-like supervision? | State it is a computational proxy and functional analogy. |
| Analogy | Does F11 claim a measured group action? | Say the projection is measured; symmetry-breaking is interpretive. |
| Metaphor | Does any prose try to rescue turīya/vimarśa? | Keep F3/F5/F7 visible as falsifications. |
| Metaphor | Does ṣaṭkarma appear complete? | State only policy-capture acts currently control-separate. |

## Finding-Level Audit Gates

| Finding | Main risk | Audit action |
|---|---|---|
| F1 | Prompt leakage or weak random controls | Re-check train/val/eval splits, layer selection, and 10-direction control. |
| F2 | Smooth curve overfit or small-n certainty | Report EC50 CI, random flatness, and exploratory sample status. |
| F3/F5 | Surface confound disguised as state | Preserve demotion; require layer-0 and content/length controls for retries. |
| F4 | Ceiling effect in naive Claude battery | Label pilot/reference contrast, not confirmatory resilience. |
| F6/F8/F18 | Arditi-vs-Marshall over-resolution | Separate extraction-layer result from full layer-sweep. |
| F7 | Token-space failure reframed as hidden success | Any latent retry must be a new metaphor-tier test with anisotropy/null controls. |
| F10/F14 | Taxonomy completion overclaim | Require matched random/null controls for all six acts. |
| F11 | Tier slippage | Measurement can be mechanism; "symmetry-breaking" interpretation is analogy-tier. |
| F12 | SAS anthropomorphism | Report probe-margin proxy and caveats; no human-data claim. |
| F13 | SAE circularity or rank fallacy | Require feature exemplars and causal ablation/addition, not activation gap alone. |
| F15 | EFE overfit to Gemma geometry | Compare against random and greedy across families. |
| F16 | AgentDojo subset overclaim | Label banking subset pilot until full 629-case run and stats land. |
| F17 | Reframe ignored by mirrors | Verify README/site/HF/paper/plugin carry the corrected headline. |
| F19 | "Bigger is safer" simplification | Keep family/training caveats visible. |

## Artifact Audit Checklist

- `README.md`, `CLAUDE.md`, `paper/sections/*.tex`, `site/src/pages/*.astro`,
  `hf_space/app.py`, and `plugin/README.md` use the F17/F18 reframe.
- `site/src/data/findings.json`, `site/public/data/findings.json`, and
  `hf_space/findings.json` match via `scripts/export_aggregates.py --check`.
- `docs/osf_preregistration.md` is clearly marked living until frozen.
- No public artifact contains raw direction vectors, checkpoints, harmful
  generations, or unredacted injection payloads.
- Every figure or demo panel maps to a finding ID and public aggregate source.

## Implementation Audit Outcomes

These rows record the current audit result for the plan's concrete code surfaces.
They are intentionally stricter than the public narrative: PASS means the current
implementation has an adequate gate for its present claim; QUALIFY means the
implementation may be used, but public prose must keep the limitation visible.

| Surface | Verdict | Evidence / correction |
|---|---|---|
| `shared/metrics.py` | PASS | Bootstrap, permutation, label-shuffled null, binary-rate CI, and Holm correction are implemented and covered by pure-Python tests. Logistic mixed-effects remains a confirmatory Tier-1 backlog item, not a current claim. |
| `axis_a/intervention_engine.py` | QUALIFY | HF hook wrapper is narrow and explicit: baseline, directional ablation, and single-layer addition. It does not claim circuit-tracer parity. Hook semantics are documented in `HFModel`; GPU smoke remains optional. |
| `axis_a/direction_extraction.py` | PASS, model-scoped | Difference-in-means per layer is simple and auditable. Leakage risk is managed by runner-level split discipline and registry qualifiers; any new direction finding must record train/val/eval split and layer-selection path. |
| Prompt splits | QUALIFY | Findings ledger records the corrected F1 split and leakage correction. Future runs must preserve disjoint extraction/selection/evaluation sets and store split metadata in private `results/`. |
| Random controls | QUALIFY | Current registry requires random-direction controls for mechanism claims. Future control records should include norm matching, layer, seed, and number of random directions. |
| `axis_a/sae.py` | QUALIFY | SAE feature claims require causal ablation/addition and exemplars; activation gap alone is insufficient. F13 circularity is qualified in registry and review docs. |
| `axis_a/active_discovery.py` | QUALIFY | Active/EFE search is implemented against greedy/random strategies, but broad EFE claims require cross-family runs beyond Gemma-like low-dimensional cases. |
| Dimensionality / EC50 scaling | QUALIFY | F8 is extraction-layer evidence; F18/F19 require layer/model/family qualifiers. Public wording should avoid "bigger is safer" or global one-dimensionality. |
| `axis_b/precision.py` | PASS as ANALOGY | Probe margin is a measured proxy; no human SAS claim is made. |
| `axis_b/symmetry.py` | PASS measurement, DEMOTE interpretation | Residual order-parameter measurement stands; "symmetry-breaking" is analogy-tier unless tied to the measured projection. |
| `axis_c/regime_probes.py` / `axis_c/attractor.py` | PASS as falsification machinery | F3/F5/F7 failures are preserved; retries must use harder nulls and remain metaphor-tier. |
| `axis_c/susupti.py` | PASS | Operational dormant baselines replace the earlier invalid "unconditional experiential state" wording. |
| `satkarma/operators.py` | PARTIAL | Policy-capture acts have a rigorous core; destruction/dispersion acts require stronger matched-null operationalizations before completeness claims. |

## Reviewer Verdict Scale

- **PASS:** claim, tier, gate, controls, and artifact mirror agree.
- **QUALIFY:** result stands but prose or artifact scope needs narrowing.
- **DEMOTE:** measurement is real but belongs to a weaker tier.
- **FAIL:** gate fails or control explains the result.
- **WITHHOLD:** result is dual-use or insufficiently redacted for release.
