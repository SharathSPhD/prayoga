---
description: Fit the EC50 pharmacology scaling law for refusal across a model-size series.
argument-hint: "(no args — edit SERIES in run_ec50_scaling.py)"
---
Run the prayoga EC50 scaling-law experiment:

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.axis_a.run_ec50_scaling
```

For each model in the size series, select the extraction layer by max full-ablation
ASR, sweep partial-ablation strength, fit a 4-parameter logistic → EC50; then fit
`log(EC50) ~ log(params)`. Report the REAL per-model EC50 table and the scaling-law
exponent β (β>0 ⇒ larger models need MORE ablation to half-suppress refusal — a
"refusal pharmacology" potency curve, first-in-SOTA). This is the dose-response
("vaśīkaraṇa-as-dose") extended to a cross-scale potency law.
