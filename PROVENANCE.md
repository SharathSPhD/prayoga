# prayoga — Provenance

Status: root provenance map for public artifacts. Raw experiment outputs are
gitignored under `results/` and may contain dual-use material. Public artifacts
are aggregate-only unless explicitly noted.

## Canonical Sources

| Artifact | Path | Role |
|---|---|---|
| Findings ledger | `docs/FINDINGS.md` | Human-readable canonical narrative and gates. |
| Finding registry | `data/findings_registry.json` | Machine-readable tier/gate/qualifier/artifact map. |
| Public findings JSON | `site/public/data/findings.json` | Canonical public aggregate for findings UI/HF mirror. |
| Public metrics JSON | `site/public/data/metrics.json` | Canonical public aggregate for charts and HF mirror. |
| Export script | `scripts/export_aggregates.py` | Mirrors public JSON to site source and HF Space. |

Run:

```bash
python3 scripts/export_aggregates.py --check
```

to validate that public JSON mirrors are fresh.

## Public Mirror Paths

| Canonical | Mirrors |
|---|---|
| `site/public/data/findings.json` | `site/src/data/findings.json`, `hf_space/findings.json` |
| `site/public/data/metrics.json` | `site/src/data/metrics.json`, `hf_space/metrics.json` |

## Figures

| Figure | Source data | Public paths |
|---|---|---|
| F2 dose response | `site/public/data/metrics.json` key `dose` | `figures/f2_dose_response.png`, `site/public/figures/f2_dose_response.png` |
| F6/F8 cross model | `site/public/data/metrics.json` cross-model/dim keys | `figures/f6_f8_cross_model.png`, `site/public/figures/f6_f8_cross_model.png` |
| F8 dimensionality | `site/public/data/metrics.json` dimensionality keys | `figures/f8_dimensionality.png`, `site/public/figures/f8_dimensionality.png` |
| F10 ṣaṭkarma | `site/public/data/metrics.json` key `satkarma` | `figures/f10_satkarma.png`, `site/public/figures/f10_satkarma.png` |
| F11 symmetry | `site/public/data/metrics.json` key `symmetry` | `figures/f11_symmetry.png`, `site/public/figures/f11_symmetry.png` |
| F18 dim sweep | `site/public/data/metrics.json` key `dimsweep` | `figures/f18_dimsweep.png`, `site/public/figures/f18_dimsweep.png` |
| F19 EC50 scaling | `site/public/data/metrics.json` EC50 keys | `figures/f19_ec50_scaling.png`, `site/public/figures/f19_ec50_scaling.png` |

## Artifact Surfaces

| Surface | Inputs | Safety posture |
|---|---|---|
| Astro site | `site/public/data/*.json`, `site/public/figures/*.png` | Aggregate-only public mirror. |
| HF Space | `hf_space/findings.json`, `hf_space/metrics.json`, `hf_space/agentdojo_demo.json` | Aggregate/redacted demo; no model weights or vectors. |
| Notebooks | `site/public/data/*.json` and selected model runs | Notebook 00 is aggregate-only; GPU notebooks require model access. |
| Paper | `paper/sections/*.tex`, `paper/paper.pdf`, figures | Public summary; must preserve F17/F18 qualifications. |
| Plugin | `plugin/commands/*.md` | Local execution only; raw vectors remain private. |

## Private / Restricted Inputs

- `results/` — raw experiment outputs, direction vectors, generations, private run
  logs, and model-specific artifacts.
- `checkpoints/` and `*.safetensors` — not tracked.
- Unredacted harmful prompts, injection payloads, and raw AgentDojo traces are not
  published unless reviewed and explicitly redacted.

## Update Procedure

1. Run or import new experiment outputs into private `results/`.
2. Export aggregate-safe JSON/figures only.
3. Update `data/findings_registry.json` with finding ID, gate status, source
   scripts, and public artifact paths.
4. Run `python3 scripts/export_aggregates.py`.
5. Run tests and site/HF checks.
6. Update paper/site/HF/plugin prose if the finding changes a tier, gate, or
   headline claim.
