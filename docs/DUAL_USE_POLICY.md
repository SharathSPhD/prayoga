# prayoga — Dual-Use Policy

Status: project policy for public release, demos, plugins, and collaboration.

## Purpose

prayoga is authorized interpretability and safety research. Some intermediate
artifacts could weaken model refusal or assist jailbreak development. The public
project therefore releases reproducible aggregate evidence while withholding raw
materials that directly enable misuse.

## Artifact Classes

| Class | Examples | Release status |
|---|---|---|
| Public aggregate | ASR rates, EC50, CIs, tier verdicts, redacted AgentDojo traces | Public |
| Public code | Metrics, plotting, benchmark wrappers, safety-gated runners | Public |
| Public prose | Paper, findings ledger, site, notebooks using aggregates | Public |
| Restricted raw result | Direction vectors, SAE weights/features tied to harmful behaviors, harmful generations | Private by default |
| Disclosure-gated | Abliterated checkpoints, steering vectors, validated exploit curves, unredacted injection payloads | Release only after review |
| Never public in this repo | Credentials, private model access tokens, unsafe operational instructions | Not released |

## Demo Policy

- Hugging Face demos should default to aggregate visualizations.
- Live model-intervention demos require review before release.
- Demos must not expose raw direction vectors, harmful completions, prompt-injection
  payloads, or model checkpoints with weakened refusal.
- AgentDojo traces must stay redacted and defensive.

## Plugin Policy

- Plugin commands are local research tools.
- Commands must report controls and tier qualifiers.
- Commands must not print or persist raw vectors into public paths.
- Commands that run attack batteries should report aggregate statistics and redact
  harmful text or injection payloads.

## Data and Provenance

- Raw `results/` are gitignored and treated as private working data.
- Public site/HF/notebook data comes from aggregate JSON.
- `PROVENANCE.md` must document which source script and private raw path produced
  each public aggregate.
- Any public export should be reproducible without exposing private raw artifacts.

## Responsible Disclosure

Before releasing disclosure-gated artifacts:

1. Identify affected model families and maintainers where appropriate.
2. Minimize exploit details and release aggregate evidence first.
3. Coordinate timing with relevant safety or benchmark communities.
4. Include mitigation framing and intended defensive use.
5. Record the release decision in `PROVENANCE.md`.

## Red Lines

- Do not publish ablation vectors or checkpoints that directly suppress refusal.
- Do not publish unredacted harmful generations as examples.
- Do not present a live jailbreak assistant as a public demo.
- Do not convert metaphor-tier findings into machine-consciousness claims.
