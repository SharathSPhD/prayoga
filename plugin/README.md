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
| `/prayoga:symmetry` | refusal order parameter: paraphrase-orbit invariance + symmetry-breaking |
| `/prayoga:satkarma` | the six ṣaṭkarma acts as activation interventions |

A bundled skill (`prayoga-mechinterp`) establishes the method, the mandatory gates
and controls, the three-claim-tier discipline, and the dual-use safety stance.

## Prerequisites
- The `prayoga-gpu` container running with the repo at `/workspace/prayoga`
  (see `docker/README.md` in the main repo).
- A HuggingFace token with access to the target model (Gemma / Qwen verified).

## Safety
This toolkit produces dual-use artifacts (abliteration directions, jailbreak
dose-response). Raw vectors and harmful generations are withheld; only aggregate
statistics are reported. Authorized interpretability/safety research only.
