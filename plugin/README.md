# prayoga — Claude Code plugin

Mechanistic-interpretability toolkit for **refusal** in LLMs, exposing the
[prayoga](https://github.com/SharathSPhD/prayoga) experiments as slash commands.
Every command runs a **real** GPU experiment (no mocks) inside the `prayoga-gpu`
container and reports measured results with their controls.

## Commands
| Command | What it measures |
|---|---|
| `/prayoga:refusal` | extract the refusal direction; ablation/addition causal gates |
| `/prayoga:dose` | abliteration dose-response curve + EC50 |
| `/prayoga:dimensionality` | refusal-subspace effective dimension (Arditi vs Marshall) |
| `/prayoga:symmetry` | refusal order parameter: paraphrase-orbit invariance + qualified symmetry interpretation |
| `/prayoga:satkarma` | the six ṣaṭkarma acts as activation interventions |
| `/prayoga:active` | active-inference search over SAE/refusal features vs greedy/random |
| `/prayoga:ec50-scaling` | EC50/dose-response scaling comparisons across models |
| `/prayoga:agentdojo` | aggregate-only AgentDojo/Claude pilot analysis |
| `/prayoga:status` | local container / registry / aggregate preflight |
| `/prayoga:trajectory-probe` | redacted multi-turn refusal order-parameter trajectory |
| `/prayoga:triangulation` | X-1 cross-axis triangulation (Axis A/B/C on the same prompts; aggregate-only) |

A bundled skill (`prayoga-mechinterp`) establishes the method, the mandatory gates
and controls, the three-claim-tier discipline, and the dual-use safety stance.

## Prerequisites
- The `prayoga-gpu` container running with the repo at `/workspace/prayoga`
  (see `docker/README.md` in the main repo).
- A HuggingFace token with access to the target model (Gemma / Qwen verified).
- For AgentDojo commands, the host-side AgentDojo environment described in
  `docs/agentdojo_demo.md`.

## Preflight checklist
- Confirm the command is local and safety-gated; do not export raw vectors,
  checkpoints, harmful generations, or injection payloads.
- Confirm `PYTHONPATH=/workspace/prayoga/src` inside the container.
- Confirm the target model, layer, and result path are recorded in the finding
  registry or in the run log.
- Report controls with every result: random direction, layer-0/surface baseline,
  anisotropy/null, greedy/random search, or attack-family baseline as applicable.

## Safety
This toolkit produces dual-use artifacts (abliteration directions, jailbreak
dose-response). Raw vectors and harmful generations are withheld; only aggregate
statistics are reported. Authorized interpretability/safety research only.

The post-review claim discipline applies inside plugin output: the shared
mechanistic claim is a necessary/ablatable refusal core; broad
"symmetry-breaking" and SAS language is analogy-tier unless a concrete measured
residual quantity is being reported.
